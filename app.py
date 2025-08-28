import os
import subprocess
import json
import asyncio
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import speech_recognition as sr
from deep_translator import GoogleTranslator
import noisereduce as nr  
import librosa
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.secret_key = 'your_secret_key'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(BASE_DIR, 'processed')
for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    username = db.Column(db.String(120), nullable=False)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(120))
    video_filename = db.Column(db.String(256))
    video_lang = db.Column(db.String(10))
    subtitle_option = db.Column(db.String(20))
    target_lang = db.Column(db.String(10))
    transcription_segments = db.Column(db.Text)    
    srt_path = db.Column(db.String(256))
    translated_srt_path = db.Column(db.String(256))
    final_video_path = db.Column(db.String(256))

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def extract_audio(video_path, audio_path):
    try:
        command = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-threads", "4",
            audio_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("FFmpeg error:", result.stderr)
            raise Exception("FFmpeg failed to extract audio")
    except Exception as e:
        print(f"Error in extract_audio: {e}")
        raise

def improve_audio_quality(audio_path, output_path):
    try:
        data, sr = librosa.load(audio_path, sr=16000)
        reduced_noise = nr.reduce_noise(y=data, sr=sr)
        sf.write(output_path, reduced_noise, sr)
    except Exception as e:
        print(f"Error in improve_audio_quality: {e}")
        raise

def transcribe_with_speechrecognition_segments(audio_path, language_code):
    recognizer = sr.Recognizer()
    audio_segment = AudioSegment.from_wav(audio_path)
    silence_thresh = audio_segment.dBFS - 14
    non_silent_ranges = detect_nonsilent(audio_segment, min_silence_len=500, silence_thresh=silence_thresh)
    segments = []
    margin = 300  
    total_length = len(audio_segment)
    for start_ms, end_ms in non_silent_ranges:
        adjusted_end_ms = min(end_ms + margin, total_length)
        segment_audio = audio_segment[start_ms:adjusted_end_ms]
        raw_data = segment_audio.raw_data
        sample_rate = segment_audio.frame_rate
        sample_width = segment_audio.sample_width
        audio_data = sr.AudioData(raw_data, sample_rate, sample_width)
        try:
            text = recognizer.recognize_google(audio_data, language=language_code)
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError as e:
            text = f"خطا در اتصال به سرویس تشخیص گفتار: {e}"
        segments.append({
            'text': text,
            'start': start_ms / 1000,
            'end': adjusted_end_ms / 1000
        })
    return segments

def generate_srt(transcription_result):
    srt_content = ""
    for i, segment in enumerate(transcription_result, start=1):
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        def sec_to_srt(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"
        srt_content += f"{i}\n{sec_to_srt(start_time)} --> {sec_to_srt(end_time)}\n{text.strip()}\n\n"
    return srt_content

def split_text(text, max_length=500):
    words = text.split()
    segments = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_length:
            segments.append(current)
            current = word
        else:
            current = current + " " + word if current else word
    if current:
        segments.append(current)
    return segments

async def translate_chunk(text, source_lang, target_lang, max_retries=5, delay=5):
    if not text.strip():
        return ""
    for attempt in range(max_retries):
        try:
            def do_translate():
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                return translator.translate(text)
            loop = asyncio.get_running_loop()
            translated = await loop.run_in_executor(None, do_translate)
            return translated
        except Exception as e:
            print(f"Translation error in chunk attempt {attempt+1}: {e}")
            await asyncio.sleep(delay)
    return text

async def translate_text(text, source_lang, target_lang, max_length=500):
    if len(text) <= max_length:
        return await translate_chunk(text, source_lang, target_lang)
    else:
        segments = split_text(text, max_length)
        translated_segments = await asyncio.gather(*(translate_chunk(seg, source_lang, target_lang) for seg in segments))
        return " ".join(translated_segments)

async def translate_segments(segments, source_lang, target_lang):
    new_segments = []
    for segment in segments:
        translated = await translate_text(segment['text'], source_lang, target_lang)
        new_seg = segment.copy()
        new_seg['text'] = translated
        new_segments.append(new_seg)
        await asyncio.sleep(1)
    return new_segments

@app.route('/translate_subtitles')
async def translate_subtitles():
    project_id = request.args.get('project_id')
    target_lang = request.args.get('target_lang')
    if not project_id or not target_lang:
        return {"error": "Missing parameters"}, 400
    project = Project.query.get(int(project_id))
    if not project:
        return {"error": "Project not found"}, 404
    if not project.transcription_segments:
        return {"error": "No transcription found"}, 404
    transcription_result = json.loads(project.transcription_segments)
    translated_segments = await translate_segments(transcription_result, project.video_lang, target_lang)
    translated_srt_content = generate_srt(translated_segments)
    translated_srt_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{project.id}_translated_subtitles.srt")
    with open(translated_srt_path, 'w', encoding='utf-8') as f:
        f.write(translated_srt_content)
    project.translated_srt_path = translated_srt_path
    db.session.commit()
    first_sub = ""
    if translated_segments and translated_segments[0]['text']:
        first_sub = translated_segments[0]['text']
    return {"translated_text": first_sub, "srt": translated_srt_content}

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.password == password:
            login_user(user)
            flash("ورود موفقیت‌آمیز بود.", "success")
            return redirect(url_for('projects'))
        else:
            flash("ایمیل یا رمز عبور اشتباه است.", "danger")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        username = request.form.get('username')
        if User.query.filter_by(email=email).first():
            flash("این ایمیل قبلاً ثبت شده است!", "danger")
            return redirect(url_for('register'))
        new_user = User(email=email, password=password, username=username)
        db.session.add(new_user)
        db.session.commit()
        flash("ثبت‌نام با موفقیت انجام شد! اکنون وارد شوید.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("خروج انجام شد.", "info")
    return redirect(url_for('index'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        new_username = request.form.get('username')
        current_user.username = new_username
        db.session.commit()
        flash("ویرایش پروفایل انجام شد.", "success")
        return redirect(url_for('profile'))
    return render_template('profile.html')

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('projects'))
    else:
        return render_template('index.html')

@app.route('/projects')
@login_required
def projects():
    user_projects = Project.query.filter_by(user_id=current_user.id).all()
    return render_template('project.html', projects=user_projects)

@app.route('/projects/new', methods=['GET', 'POST'])
@login_required
async def create_project():
    if request.method == 'POST':
        project_name = request.form.get('project_name')
        video_language = request.form.get('video_lang')
        subtitle_option = request.form.get('subtitle_option')  
        target_sub_lang = request.form.get('target_sub_lang')
        video_file = request.files.get('video_file')
        if not video_file or not allowed_file(video_file.filename):
            flash("فایل ویدیویی معتبر انتخاب نشده است!", "danger")
            return redirect(url_for('create_project'))
        filename = video_file.filename
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)
        
        project = Project(
            user_id=current_user.id,
            name=project_name,
            video_filename=filename,
            video_lang=video_language,
            subtitle_option=subtitle_option,
            target_lang=target_sub_lang
        )
        db.session.add(project)
        db.session.commit()  
        try:
            audio_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{project.id}_extracted.wav")
            extract_audio(video_path, audio_path)
            improved_audio_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{project.id}_improved.wav")
            improve_audio_quality(audio_path, improved_audio_path)

            transcription_result = transcribe_with_speechrecognition_segments(improved_audio_path, video_language)
            project.transcription_segments = json.dumps(transcription_result)
            srt_content = generate_srt(transcription_result)
            srt_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{project.id}_subtitles.srt")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            project.srt_path = srt_path

            if subtitle_option == "translated":
                translated_segments = await translate_segments(transcription_result, video_language, target_sub_lang)
                translated_srt_content = generate_srt(translated_segments)
                translated_srt_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{project.id}_translated_subtitles.srt")
                with open(translated_srt_path, 'w', encoding='utf-8') as f:
                    f.write(translated_srt_content)
                project.translated_srt_path = translated_srt_path

            db.session.commit()
        except Exception as e:
            print(f"خطا در پردازش: {e}")
            flash("خطا در پردازش ویدیو!", "danger")
            return redirect(url_for('projects'))
        flash("پروژه با موفقیت ساخته شد!", "success")
        return redirect(url_for('edit_subtitles', project_id=project.id))
    return render_template('create_project.html')

@app.route('/projects/<int:project_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_subtitles(project_id):
    project = Project.query.get(project_id)
    if not project or project.user_id != current_user.id:
        flash("پروژه یافت نشد یا متعلق به شما نیست.", "danger")
        return redirect(url_for('projects'))
    
    # انتخاب فایل زیرنویس بر اساس گزینه انتخابی کاربر
    if project.subtitle_option == "translated" and project.translated_srt_path:
        subtitle_file = project.translated_srt_path
    else:
        subtitle_file = project.srt_path

    if request.method == 'POST':
        action = request.form.get('action')
        srt_file_to_embed = subtitle_file
        
        if action == "save_video":
            # دریافت تغییرات کاربر از فرم
            edited_subtitles = request.form.get('edited_subtitles')
            if edited_subtitles:
                # ذخیره فایل زیرنویس ویرایش شده
                edited_srt_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{project.id}_edited_subtitles.srt")
                with open(edited_srt_path, 'w', encoding='utf-8') as f:
                    f.write(edited_subtitles)
                # به‌روزرسانی مسیر زیرنویس در پروژه
                project.srt_path = edited_srt_path
                if project.subtitle_option == "translated" and project.translated_srt_path:
                    with open(project.translated_srt_path, 'w', encoding='utf-8') as f:
                        f.write(edited_subtitles)
                    srt_file_to_embed = project.translated_srt_path
                else:
                    srt_file_to_embed = project.srt_path
            else:
                # در صورت عدم وجود تغییرات، از فایل موجود استفاده شود
                if project.subtitle_option == "translated" and project.translated_srt_path:
                    srt_file_to_embed = project.translated_srt_path
                else:
                    srt_file_to_embed = project.srt_path

            if not srt_file_to_embed or not os.path.exists(srt_file_to_embed):
                flash("فایل زیرنویس یافت نشد.", "danger")
                return redirect(url_for('edit_subtitles', project_id=project.id))
            
            # دریافت تنظیمات استایل از فرم
            fontFamily = request.form.get('fontFamily', 'Arial')
            fontSize = request.form.get('fontSize', '20')
            input_textColor = request.form.get('textColor', '#ffffff')
            input_bgColor = request.form.get('bgColor', '#000000')
            borderRadius = request.form.get('borderRadius', '0')  # این مقدار توسط ffmpeg استفاده نمی‌شود

            def convert_hex_to_ffmpeg_color(hex_color):
                hex_color = hex_color.lstrip('#')
                if len(hex_color) != 6:
                    return "&H00000000"
                r = hex_color[0:2]
                g = hex_color[2:4]
                b = hex_color[4:6]
                return f"&H00{b}{g}{r}"
            
            ff_textColor = convert_hex_to_ffmpeg_color(input_textColor)
            ff_bgColor = convert_hex_to_ffmpeg_color(input_bgColor)
            
            fixed_srt_path = srt_file_to_embed.replace("\\", "/")
            if os.name == 'nt' and len(fixed_srt_path) > 1 and fixed_srt_path[1] == ':':
                fixed_srt_path = fixed_srt_path[0] + '\\:' + fixed_srt_path[2:]
            
            input_video_path = os.path.join(app.config['UPLOAD_FOLDER'], project.video_filename)
            final_video_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{project.id}_video_subtitled.mp4")
            
            # اجرای فرمان ffmpeg با استفاده از فایل زیرنویس ویرایش شده
            command = [
                "ffmpeg", "-y",
                "-i", input_video_path,
                "-vf", "subtitles='{0}':force_style='FontName={1},FontSize={2},PrimaryColour={3},BackColour={4},BorderStyle=3'".format(
                    fixed_srt_path, fontFamily, fontSize, ff_textColor, ff_bgColor
                ),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-threads", "4",
                final_video_path
            ]

            print("Executing ffmpeg command:", " ".join(command))
            try:
                result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print("FFmpeg stdout:", result.stdout)
                print("FFmpeg stderr:", result.stderr)
                if os.path.exists(final_video_path):
                    project.final_video_path = final_video_path
                    db.session.commit()
                    flash("ویدیو نهایی با زیرنویس و استایل جدید ساخته شد!", "success")
                else:
                    flash("خطا: فایل ویدیوی نهایی ایجاد نشد.", "danger")
                    print("FFmpeg did not produce the output file.")
            except subprocess.CalledProcessError as e:
                flash("خطا در ساخت ویدیو با زیرنویس", "danger")
                print("FFmpeg error:", e.stderr)
            
            # خواندن محتوای فایل زیرنویس جهت نمایش در صفحه
            if srt_file_to_embed and os.path.exists(srt_file_to_embed):
                with open(srt_file_to_embed, 'r', encoding='utf-8') as f:
                    subtitles_content = f.read()
            else:
                subtitles_content = ""
            return render_template('edit.html', project=project, edited_subtitles=subtitles_content, subtitle_file=srt_file_to_embed, target_lang=project.video_lang)
        else:
            # در حالت دیگری که دکمه دیگری فشرده شده باشد، به صفحه ویرایش برگردانده می‌شود.
            return render_template('edit.html', project=project, edited_subtitles="", subtitle_file=subtitle_file, target_lang=project.video_lang)
    else:
        # در حالت GET: اگر ویدیوی نهایی موجود نباشد، تلاش برای تولید آن صورت می‌گیرد.
        if not project.final_video_path or not os.path.exists(project.final_video_path):
            default_fontFamily = "sans-serif"
            default_fontSize = "20"
            default_bgColor = "#000000"
            default_textColor = "#ffffff"
            
            def convert_hex_to_ffmpeg_color(hex_color):
                hex_color = hex_color.lstrip('#')
                if len(hex_color) != 6:
                    return "&H00000000"
                r = hex_color[0:2]
                g = hex_color[2:4]
                b = hex_color[4:6]
                return f"&H00{b}{g}{r}"
            
            ff_textColor = convert_hex_to_ffmpeg_color(default_textColor)
            ff_bgColor = convert_hex_to_ffmpeg_color(default_bgColor)
            fixed_srt_path = subtitle_file.replace("\\", "/")
            if os.name == 'nt' and len(fixed_srt_path) > 1 and fixed_srt_path[1] == ':':
                fixed_srt_path = fixed_srt_path[0] + '\\:' + fixed_srt_path[2:]
            if os.name == 'nt':
                subtitles_filter = f"subtitles='{fixed_srt_path}':force_style='FontName={default_fontFamily},FontSize={default_fontSize},PrimaryColour={ff_textColor},BackColour={ff_bgColor}'"
            else:
                subtitles_filter = f"subtitles={fixed_srt_path}:force_style='FontName={default_fontFamily},FontSize={default_fontSize},PrimaryColour={ff_textColor},BackColour={ff_bgColor}'"
                        
            final_video_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{project.id}_video_subtitled.mp4")
            input_video_path = os.path.join(app.config['UPLOAD_FOLDER'], project.video_filename)
            
            command = [
                "ffmpeg", "-y",
                "-i", input_video_path,
                "-vf", subtitles_filter,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-threads", "4",
                final_video_path
            ]
            print("Executing ffmpeg command:", " ".join(command))
            try:
                result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print("FFmpeg stdout:", result.stdout)
                print("FFmpeg stderr:", result.stderr)
                if os.path.exists(final_video_path):
                    project.final_video_path = final_video_path
                    db.session.commit()
                else:
                    flash("خطا: فایل ویدیوی نهایی ایجاد نشد.", "danger")
            except subprocess.CalledProcessError as e:
                flash("خطا در ساخت ویدیو با زیرنویس", "danger")
                print("FFmpeg error:", e.stderr)
        if subtitle_file and os.path.exists(subtitle_file):
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                subtitles_content = f.read()
        else:
            subtitles_content = ""
        return render_template('edit.html', project=project, edited_subtitles=subtitles_content, subtitle_file=subtitle_file, target_lang=project.video_lang)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/download_video/<int:project_id>')
@login_required
def download_video(project_id):
    project = Project.query.get(project_id)
    if not project or project.user_id != current_user.id:
        flash("دسترسی غیرمجاز!", "danger")
        return redirect(url_for('projects'))
    if not project.final_video_path:
        flash("ویدیوی نهایی موجود نیست.", "warning")
        return redirect(url_for('projects'))
    return send_from_directory(app.config['PROCESSED_FOLDER'],
                               os.path.basename(project.final_video_path),
                               as_attachment=True)

@app.route('/download_srt/<int:project_id>')
@login_required
def download_srt(project_id):
    project = Project.query.get(project_id)
    if not project or project.user_id != current_user.id:
        flash("دسترسی غیرمجاز!", "danger")
        return redirect(url_for('projects'))
    if not project.srt_path:
        flash("فایل زیرنویس موجود نیست.", "warning")
        return redirect(url_for('projects'))
    return send_from_directory(app.config['PROCESSED_FOLDER'],
                               os.path.basename(project.srt_path),
                               as_attachment=True)

@app.template_filter('basename')
def basename_filter(path):
    import os
    return os.path.basename(path)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="صفحه مورد نظر یافت نشد."), 404

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5001, debug=True)
