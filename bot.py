import os
import json
import asyncio
import functools
import logging
import cv2
import srt
from sqlalchemy import Column, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, InputFile
from telegram.ext import (ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
                          ConversationHandler, ContextTypes, filters)
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import speech_recognition as sr
from deep_translator import GoogleTranslator
import noisereduce as nr
import librosa
import soundfile as sf

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO)
Base = declarative_base()
DB = "sqlite:///db.sqlite3"
eng = create_engine(DB, connect_args={"check_same_thread": False})
Ses = sessionmaker(bind=eng)
ses = Ses()

def to_thread(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

class U(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    tid = Column(String(50), unique=True, nullable=False)
    un = Column(String(50), nullable=False)

class P(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    uid = Column(String(50), nullable=False)
    nm = Column(String(120))
    vf = Column(String(256))
    vl = Column(String(10))
    so = Column(String(20))
    tl = Column(String(10))
    ts = Column(Text)
    sp = Column(String(256))
    tsp = Column(String(256))
    fp = Column(String(256))

Base.metadata.create_all(eng)

BOT_TOKEN = "8093363508:AAHnJc2WjLQ5pM7QZig-MxpHVbcTbAM7DgQ"
UP_DIR = os.path.join(os.getcwd(), "up")
PR_DIR = os.path.join(os.getcwd(), "pr")
for d in [UP_DIR, PR_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

PN, VL, SO, TL, VF = range(1, 6)
C_MAIN, C_FONT, C_SIZE, C_TEXT, C_BG, C_TIME, C_EDIT, C_CONFIRM = range(10, 18)

def prepare_subtitle_path(srt_path):
    return os.path.normpath(srt_path)

def conv_color(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0, 0, 0)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV از BGR استفاده می‌کند

def kb_lang():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🇺🇸 English (en)", callback_data="lang_en"),
         InlineKeyboardButton("🇮🇷 فارسی (fa)", callback_data="lang_fa")],
        [InlineKeyboardButton("🇸🇦 عربی (ar)", callback_data="lang_ar"),
         InlineKeyboardButton("🔙 بازگشت", callback_data="cancel")]
    ])

# ----------------- توابع ترجمه -----------------
async def trc(tx, sl, tl, mr=5, d=5):
    if not tx.strip():
        return ""
    for attempt in range(mr):
        try:
            def do_tr():
                translator = GoogleTranslator(source=sl, target=tl)
                return translator.translate(tx)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, do_tr)
        except Exception as e:
            logging.error(f"Translation error (attempt {attempt+1}): {e}")
            await asyncio.sleep(d)
    return tx

async def tr_txt(tx, sl, tl, ml=500):
    if len(tx) <= ml:
        return await trc(tx, sl, tl)
    else:
        parts = tx.split()
        segs = []
        current = ""
        for word in parts:
            if len(current) + len(word) + 1 > ml:
                segs.append(current)
                current = word
            else:
                current = current + " " + word if current else word
        if current:
            segs.append(current)
        translations = await asyncio.gather(*(trc(seg, sl, tl) for seg in segs))
        return " ".join(translations)

async def tr_segs(segs, sl, tl):
    new_segs = []
    for seg in segs:
        translated = await tr_txt(seg['text'], sl, tl)
        nseg = seg.copy()
        nseg['text'] = translated
        new_segs.append(nseg)
        await asyncio.sleep(1)
    return new_segs

def alf(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv'}

def ext_a(vp, ap):
    clip = VideoFileClip(vp)
    clip.audio.write_audiofile(ap, logger=None)

def imp_a(ap, op):
    data, sr_rate = librosa.load(ap, sr=16000)
    reduced = nr.reduce_noise(y=data, sr=sr_rate)
    sf.write(op, reduced, sr_rate)

def ts_seg(ap, lc):
    rec = sr.Recognizer()
    aud = AudioSegment.from_wav(ap)
    thresh = aud.dBFS - 14
    nonsilent = detect_nonsilent(aud, min_silence_len=500, silence_thresh=thresh)
    segs = []
    margin = 300
    total = len(aud)
    for start, end in nonsilent:
        ae = min(end + margin, total)
        segment = aud[start:ae]
        raw = segment.raw_data
        srn = segment.frame_rate
        sw = segment.sample_width
        audio_data = sr.AudioData(raw, srn, sw)
        try:
            text = rec.recognize_google(audio_data, language=lc)
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError as ex:
            text = f"❌ خطا: {ex}"
        segs.append({'text': text, 'start': start/1000, 'end': ae/1000})
    return segs

def gen_srt(tsr):
    srt_text = ""
    for i, seg in enumerate(tsr, start=1):
        st = seg['start']
        et = seg['end']
        tx = seg['text']
        def conv(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            ms = int((t - int(t)) * 1000)
            return f"{h:02}:{m:02}:{s:02},{ms:03}"
        srt_text += f"{i}\n{conv(st)} --> {conv(et)}\n{tx.strip()}\n\n"
    return srt_text

def adjust_srt_timing(srt_path, offset, out_path):
    def adjust_time(t_str, offset):
        h, m, s_ms = t_str.split(":")
        s, ms = s_ms.split(",")
        total = int(h)*3600 + int(m)*60 + int(s) + float(f"0.{ms}") + offset
        h_new = int(total // 3600)
        m_new = int((total % 3600) // 60)
        s_new = int(total % 60)
        ms_new = int(round((total - int(total)) * 1000))
        return f"{h_new:02}:{m_new:02}:{s_new:02},{ms_new:03}"
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if "-->" in line:
            parts = line.strip().split(" --> ")
            if len(parts) == 2:
                new_start = adjust_time(parts[0], offset)
                new_end = adjust_time(parts[1], offset)
                new_lines.append(f"{new_start} --> {new_end}\n")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    return out_path

def kb_start():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 شروع ساخت پروژه", callback_data="new")],
        [InlineKeyboardButton("ℹ️ راهنما", callback_data="help")],
        [InlineKeyboardButton("🗂 تاریخچه پروژه‌ها", callback_data="history")]
    ])

def kb_cancel():
    return InlineKeyboardMarkup([[InlineKeyboardButton("❌ لغو", callback_data="cancel")]])

def kb_so():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📄 زیرنویس اصلی", callback_data="o"),
         InlineKeyboardButton("🌐 ترجمه زیرنویس", callback_data="t")],
        [InlineKeyboardButton("❌ لغو", callback_data="cancel")]
    ])

def kb_main():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🏠 منوی اصلی", callback_data="main")]])

def kb_custom_main():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🖋️ فونت", callback_data="c_font"),
         InlineKeyboardButton("🔢 اندازه فونت", callback_data="c_size")],
        [InlineKeyboardButton("🎨 رنگ متن", callback_data="c_text"),
         InlineKeyboardButton("🖌️ رنگ پس‌زمینه", callback_data="c_bg")],
        [InlineKeyboardButton("⏱️ تنظیم زمان", callback_data="c_time")],
        [InlineKeyboardButton("✏️ ویرایش زیرنویس", callback_data="c_edit")],
        [InlineKeyboardButton("💾 ذخیره تغییرات", callback_data="c_save")],
        [InlineKeyboardButton("❌ لغو", callback_data="c_cancel")]
    ])

def kb_font():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("Arial", callback_data="font_Arial"),
         InlineKeyboardButton("Helvetica", callback_data="font_Helvetica")],
        [InlineKeyboardButton("Times New Roman", callback_data="font_TNR"),
         InlineKeyboardButton("Courier New", callback_data="font_Courier")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="c_back")]
    ])

def kb_size():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("16", callback_data="size_16"),
         InlineKeyboardButton("20", callback_data="size_20")],
        [InlineKeyboardButton("24", callback_data="size_24"),
         InlineKeyboardButton("28", callback_data="size_28")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="c_back")]
    ])

def kb_color(kind):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚪ سفید", callback_data=f"{kind}_#ffffff"),
         InlineKeyboardButton("⚫ مشکی", callback_data=f"{kind}_#000000")],
        [InlineKeyboardButton("🔴 قرمز", callback_data=f"{kind}_#ff0000"),
         InlineKeyboardButton("🟢 سبز", callback_data=f"{kind}_#00ff00")],
        [InlineKeyboardButton("🔵 آبی", callback_data=f"{kind}_#0000ff")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="c_back")]
    ])

def kb_time():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⏪ -1 ثانیه", callback_data="time_-1"),
         InlineKeyboardButton("⏪ -0.5 ثانیه", callback_data="time_-0.5")],
        [InlineKeyboardButton("⏺️ بدون تغییر", callback_data="time_0"),
         InlineKeyboardButton("⏩ +0.5 ثانیه", callback_data="time_0.5")],
        [InlineKeyboardButton("⏩ +1 ثانیه", callback_data="time_1")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="c_back")]
    ])

def parse_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    subs = list(srt.parse(content))
    subtitles = []
    for sub in subs:
        start = sub.start.total_seconds()
        end = sub.end.total_seconds()
        text = sub.content.replace("\n", " ")
        subtitles.append({'start': start, 'end': end, 'text': text})
    return subtitles

def add_subtitles_opencv(input_video, srt_path, output_video, style):
    subtitles = parse_srt(srt_path)
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur_time = frame_idx / fps
        for sub in subtitles:
            if sub['start'] <= cur_time <= sub['end']:
                text = sub['text']
                font = style.get('font', cv2.FONT_HERSHEY_SIMPLEX)
                font_scale = style.get('font_scale', 1)
                thickness = style.get('thickness', 2)
                text_color = style.get('text_color', (255, 255, 255))
                bg_color = style.get('bg_color', (0, 0, 0))
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                x = int((frame_width - text_width) / 2)
                y = frame_height - text_height - 20
                cv2.rectangle(frame, (x-5, y - text_height - 5), (x+text_width+5, y+5), bg_color, cv2.FILLED)
                cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
                break
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

# توابع مربوط به ربات تلگرام
async def select_vl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    lang_code = query.data.split("_")[1]
    ctx.user_data['vl'] = lang_code
    await query.message.edit_text(
        f"✅ زبان ویدیو انتخاب شد: {lang_code}\n📝 گزینه زیرنویس را انتخاب کنید:",
        reply_markup=kb_so()
    )
    return SO

async def select_tl(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    lang_code = query.data.split("_")[1]
    ctx.user_data['tl'] = lang_code
    await query.message.edit_text(
        f"✅ زبان مقصد انتخاب شد: {lang_code}\n📤 لطفاً فایل ویدیویی را ارسال کنید:",
        reply_markup=kb_cancel()
    )
    return VF

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سلام! 🌟 به ربات حرفه‌ای پردازش ویدیو خوش آمدید.",
        reply_markup=kb_start()
    )

async def cb_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "new":
        await query.message.edit_text("👋 لطفاً نام پروژه را وارد کنید:", reply_markup=kb_cancel())
        return PN
    elif data == "help":
        await query.message.edit_text(
            "راهنما:\n\nاین ربات برای پردازش ویدیو طراحی شده است. مراحل کار به صورت دکمه‌ای انجام می‌شود:\n"
            "1️⃣ وارد کردن نام پروژه\n2️⃣ انتخاب زبان ویدیو\n3️⃣ انتخاب نوع زیرنویس (اصلی یا ترجمه)\n"
            "4️⃣ (در صورت ترجمه) انتخاب زبان مقصد\n5️⃣ ارسال فایل ویدیویی\n\n"
            "برای لغو در هر مرحله، دکمه ❌ لغو را فشار دهید.",
            reply_markup=kb_main()
        )
        return ConversationHandler.END
    elif data == "cancel":
        await query.message.edit_text("❌ عملیات لغو شد.", reply_markup=kb_main())
        return ConversationHandler.END
    elif data == "main":
        await query.message.edit_text("🏠 بازگشت به منوی اصلی.", reply_markup=kb_start())
        return ConversationHandler.END
    elif data == "history":
        uid = str(query.from_user.id)
        projects = ses.query(P).filter_by(uid=uid).all()
        if not projects:
            await query.message.edit_text("شما تاکنون هیچ پروژه‌ای ایجاد نکرده‌اید.", reply_markup=kb_start())
            return ConversationHandler.END
        text = "📜 تاریخچه پروژه‌های شما:\n\n"
        keyboard = []
        for proj in projects:
            text += f"• پروژه: {proj.nm} (شناسه: {proj.id})\n"
            if proj.fp and os.path.exists(proj.fp):
                keyboard.append([InlineKeyboardButton(f"دانلود {proj.nm}", callback_data=f"history_download_{proj.id}")])
        reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
        await query.message.edit_text(text, reply_markup=reply_markup)
        return ConversationHandler.END
    elif data in ["o", "t"]:
        ctx.user_data['so'] = data
        if data == "t":
            await query.message.edit_text("🌐 لطفاً زبان مقصد را انتخاب کنید:", reply_markup=kb_lang())
            return TL
        else:
            await query.message.edit_text("📤 لطفاً فایل ویدیویی را ارسال کنید:", reply_markup=kb_cancel())
            return VF
    else:
        await query.answer("❌ گزینه نامعتبر انتخاب شده است.", show_alert=True)
        return ConversationHandler.END

async def in_pn(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message.text:
        await update.message.reply_text("❌ نام پروژه باید به صورت متن باشد.", reply_markup=kb_cancel())
        return PN
    ctx.user_data['pn'] = update.message.text.strip()
    await update.message.reply_text("🔤 لطفاً زبان ویدیو را انتخاب کنید:", reply_markup=kb_lang())
    return VL

async def in_vf(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message.video and not update.message.document:
        await update.message.reply_text("❌ لطفاً فقط یک فایل ویدیویی معتبر ارسال کنید.", reply_markup=kb_cancel())
        return VF
    f = update.message.video if update.message.video else update.message.document
    fn = f.file_name if hasattr(f, "file_name") else f.file_id + ".mp4"
    if not alf(fn):
        await update.message.reply_text("❌ فرمت فایل پشتیبانی نمی‌شود.", reply_markup=kb_cancel())
        return VF
    file = await f.get_file()
    fp = os.path.join(UP_DIR, fn)
    await file.download_to_drive(fp)
    uid = str(update.effective_user.id)
    u = ses.query(U).filter_by(tid=uid).first()
    if not u:
        u = U(tid=uid, un=update.effective_user.username or "no")
        ses.add(u)
        ses.commit()
    p = P(uid=uid, nm=ctx.user_data.get('pn'), vf=fn, vl=ctx.user_data.get('vl'),
          so=ctx.user_data.get('so'), tl=ctx.user_data.get('tl', ""))
    ses.add(p)
    ses.commit()
    ctx.user_data['pid'] = p.id

    msg = await update.message.reply_text("📥 ویدیو دریافت شد. در حال پردازش، لطفاً کمی صبر کنید...", reply_markup=kb_cancel())

    try:
        ap = os.path.join(PR_DIR, f"{p.id}_ex.wav")
        await msg.edit_text("🔄 استخراج صدا...")
        await to_thread(ext_a, fp, ap)

        ip = os.path.join(PR_DIR, f"{p.id}_im.wav")
        await msg.edit_text("🔄 کاهش نویز و پردازش صوت...")
        await to_thread(imp_a, ap, ip)

        await msg.edit_text("🔄 تشخیص گفتار...")
        tsr = await to_thread(ts_seg, ip, p.vl)
        p.ts = json.dumps(tsr)

        sp = os.path.join(PR_DIR, f"{p.id}_sub.srt")
        with open(sp, "w", encoding="utf-8") as f1:
            f1.write(gen_srt(tsr))
        p.sp = sp

        if p.so == "t":
            await msg.edit_text("🔄 ترجمه زیرنویس...")
            nts = await tr_segs(tsr, p.vl, p.tl)
            tsp = os.path.join(PR_DIR, f"{p.id}_trsub.srt")
            with open(tsp, "w", encoding="utf-8") as f2:
                f2.write(gen_srt(nts))
            p.tsp = tsp

        srtp = p.tsp if p.so == "t" and p.tsp else p.sp
        fixed_srt = prepare_subtitle_path(srtp)
        ivp = os.path.join(UP_DIR, p.vf)
        fpv = os.path.join(PR_DIR, f"{p.id}_final.mp4")

        await msg.edit_text("🔄 افزودن زیرنویس به ویدیو (با استفاده از OpenCV)...")
        style_options = {
            'font': cv2.FONT_HERSHEY_SIMPLEX,
            'font_scale': 1,
            'text_color': (255, 255, 255),
            'bg_color': (0, 0, 0),
            'thickness': 2
        }
        await to_thread(add_subtitles_opencv, ivp, fixed_srt, fpv, style_options)

        if not os.path.exists(fpv):
            raise Exception("❌ فایل ویدیوی نهایی ایجاد نشد.")
        p.fp = fpv
        ses.commit()
    except Exception as e:
        logging.error(f"Processing error: {e}")
        await msg.edit_text("❌ مشکلی در پردازش ویدیو رخ داده است. لطفاً دوباره تلاش کنید.", reply_markup=kb_main())
        return ConversationHandler.END

    final_kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("✏️ سفارشی‌سازی زیرنویس", callback_data="customize")],
        [InlineKeyboardButton("📥 دانلود ویدیو", callback_data="download")],
        [InlineKeyboardButton("🏠 منوی اصلی", callback_data="main")]
    ])
    await msg.edit_text("✅ پردازش ویدیو به پایان رسید. ویدیو نهایی آماده است.", reply_markup=final_kb)
    return ConversationHandler.END

# هندلر نهایی فقط برای دکمه‌های "دانلود" و "منوی اصلی"
async def final_cb_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    pid = ctx.user_data.get("pid")
    p = ses.get(P, pid) if pid else None
    if data == "main":
        await query.message.edit_text("🏠 بازگشت به منوی اصلی.", reply_markup=kb_start())
    elif data == "download":
        if p and p.fp and os.path.exists(p.fp):
            # نمایش پیام وضعیت ارسال جهت جلوگیری از کلیک مکرر
            await query.message.edit_text("⏳ در حال ارسال ویدیو، لطفاً کمی صبر کنید...", reply_markup=kb_main())
            with open(p.fp, "rb") as video_file:
                await ctx.bot.send_document(
                    chat_id=query.message.chat.id,
                    document=InputFile(video_file, filename=f"{p.id}_final.mp4"),
                    caption="🎬 ویدیو نهایی شما"
                )
        else:
            await query.message.edit_text("❌ ویدیو نهایی موجود نیست.", reply_markup=kb_main())
    else:
        await query.answer("❌ گزینه نامعتبر انتخاب شده است.", show_alert=True)
    return ConversationHandler.END

# هندلر سفارشی‌سازی
async def customize_cb_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    if data == "c_font":
        return await c_font(update, ctx)
    elif data.startswith("font_"):
        return await c_set_font(update, ctx)
    elif data == "c_size":
        return await c_size(update, ctx)
    elif data.startswith("size_"):
        return await c_set_size(update, ctx)
    elif data == "c_text":
        return await c_textcolor(update, ctx)
    elif data.startswith("text_"):
        return await c_set_textcolor(update, ctx)
    elif data == "c_bg":
        return await c_bgcolor(update, ctx)
    elif data.startswith("bg_"):
        return await c_set_bgcolor(update, ctx)
    elif data == "c_time":
        return await c_time(update, ctx)
    elif data.startswith("time_"):
        return await c_set_time(update, ctx)
    elif data == "c_edit":
        await query.message.edit_text("✏️ لطفاً متن جدید زیرنویس را ارسال کنید:")
        return C_EDIT
    elif data == "c_save":
        return await c_save(update, ctx)
    elif data == "c_cancel":
        return await c_cancel(update, ctx)
    elif data == "c_back":
        return await c_back(update, ctx)
    else:
        await query.answer("❌ گزینه نامعتبر در سفارشی‌سازی.", show_alert=True)
        return C_MAIN

async def history_download_handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    pid = data.split("_")[-1]
    project = ses.get(P, pid)
    if project and project.fp and os.path.exists(project.fp):
        with open(project.fp, "rb") as video_file:
            await query.message.reply_document(
                document=InputFile(video_file, filename=f"{project.id}_final.mp4"),
                caption=f"🎬 پروژه: {project.nm}",
                reply_markup=kb_start()
            )
    else:
        await query.message.edit_text("❌ فایل ویدیوی نهایی این پروژه یافت نشد.", reply_markup=kb_start())

async def history_command(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_user.id)
    projects = ses.query(P).filter_by(uid=uid).all()
    if not projects:
        await update.message.reply_text("شما تاکنون هیچ پروژه‌ای ایجاد نکرده‌اید.", reply_markup=kb_start())
        return
    text = "📜 تاریخچه پروژه‌های شما:\n\n"
    keyboard = []
    for proj in projects:
        text += f"• پروژه: {proj.nm} (شناسه: {proj.id})\n"
        if proj.fp and os.path.exists(proj.fp):
            keyboard.append([InlineKeyboardButton(f"دانلود {proj.nm}", callback_data=f"history_download_{proj.id}")])
    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    await update.message.reply_text(text, reply_markup=reply_markup)

async def c_back(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("🔄 بروزرسانی سفارشی‌سازی:", reply_markup=kb_custom_main())
    return C_MAIN

async def c_font(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("🖋️ لطفاً یک فونت انتخاب کنید:", reply_markup=kb_font())
    return C_FONT

async def c_set_font(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    font = query.data.split("_")[1]
    ctx.user_data['font'] = "Times New Roman" if font == "TNR" else font
    await query.message.edit_text(f"✅ فونت به {ctx.user_data['font']} تغییر یافت.", reply_markup=kb_custom_main())
    return C_MAIN

async def c_size(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("🔢 اندازه فونت را انتخاب کنید:", reply_markup=kb_size())
    return C_SIZE

async def c_set_size(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    size = query.data.split("_")[1]
    ctx.user_data['font_size'] = size
    await query.message.edit_text(f"✅ اندازه فونت به {size} تغییر یافت.", reply_markup=kb_custom_main())
    return C_MAIN

async def c_textcolor(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("🎨 رنگ متن را انتخاب کنید:", reply_markup=kb_color("text"))
    return C_TEXT

async def c_set_textcolor(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    hex_color = query.data.split("_")[1]
    ctx.user_data['text_color'] = conv_color(hex_color)
    await query.message.edit_text("✅ رنگ متن تغییر یافت.", reply_markup=kb_custom_main())
    return C_MAIN

async def c_bgcolor(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("🖌️ رنگ پس‌زمینه زیرنویس را انتخاب کنید:", reply_markup=kb_color("bg"))
    return C_BG

async def c_set_bgcolor(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    hex_color = query.data.split("_")[1]
    ctx.user_data['bg_color'] = conv_color(hex_color)
    await query.message.edit_text("✅ رنگ پس‌زمینه تغییر یافت.", reply_markup=kb_custom_main())
    return C_MAIN

async def c_time(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("⏱️ تنظیم زمان‌بندی زیرنویس:", reply_markup=kb_time())
    return C_TIME

async def c_set_time(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    try:
        offset = float(query.data.split("_")[1])
    except ValueError:
        await query.answer("❌ مقدار زمان نامعتبر است.", show_alert=True)
        return C_TIME
    ctx.user_data['time_offset'] = offset
    await query.message.edit_text(f"✅ زمان‌بندی به {offset} ثانیه تغییر یافت.", reply_markup=kb_custom_main())
    return C_MAIN

async def c_edit(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✏️ لطفاً متن جدید زیرنویس را ارسال کنید:")
    return C_EDIT

async def c_set_edit(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not update.message.text:
        await update.message.reply_text("❌ لطفاً متن معتبر وارد کنید.")
        return C_EDIT
    ctx.user_data['custom_sub'] = update.message.text
    await update.message.reply_text("✅ متن زیرنویس به صورت دستی تغییر یافت.", reply_markup=kb_custom_main())
    return C_MAIN

async def c_save(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    pid = ctx.user_data.get("pid")
    p = ses.get(P, pid) if pid else None
    if not p:
        await query.message.edit_text("❌ پروژه یافت نشد.", reply_markup=kb_main())
        return ConversationHandler.END
    srt_path = p.sp
    if ctx.user_data.get("time_offset", 0.0) != 0.0:
        new_srt = os.path.join(PR_DIR, f"{p.id}_sub_custom.srt")
        adjust_srt_timing(srt_path, ctx.user_data["time_offset"], new_srt)
        srt_path = new_srt
    if ctx.user_data.get("custom_sub"):
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(ctx.user_data["custom_sub"])
    ivp = os.path.join(UP_DIR, p.vf)
    fpv = os.path.join(PR_DIR, f"{p.id}_final.mp4")
    if p.so == "t" and p.tsp:
        srtp = p.tsp
    else:
        srtp = p.sp
    fixed_srt = os.path.normpath(srtp)

    style_options = {
        'font': cv2.FONT_HERSHEY_SIMPLEX,
        'font_scale': 1,
        'text_color': (255, 255, 255),
        'bg_color': (0, 0, 0),
        'thickness': 2
    }
    try:
        await to_thread(add_subtitles_opencv, ivp, fixed_srt, fpv, style_options)
        p.fp = fpv
        ses.commit()
        await query.message.edit_text("💾 تغییرات ذخیره شد. ویدیو نهایی سفارشی شده در حال ارسال است.", reply_markup=kb_main())
        with open(p.fp, "rb") as video_file:
            await query.message.reply_document(
                document=InputFile(video_file, filename=f"{p.id}_final.mp4"),
                caption="🎬 این ویدیو با زیرنویس سفارشی شده است.",
                reply_markup=kb_main()
            )
    except Exception as e:
        logging.error(f"Customization error: {e}")
        await query.message.edit_text("❌ خطا در اعمال تغییرات.", reply_markup=kb_main())
    return ConversationHandler.END

async def c_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.message.edit_text("❌ سفارشی‌سازی لغو شد.", reply_markup=kb_main())
    else:
        await update.message.reply_text("❌ سفارشی‌سازی لغو شد.", reply_markup=kb_main())
    return ConversationHandler.END

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error("Exception while handling an update:", exc_info=context.error)

main_conv = ConversationHandler(
    entry_points=[CallbackQueryHandler(cb_handler, pattern="^(new|help|cancel|main|o|t|history)$"),
                  CommandHandler("new", cb_handler)],
    states={
        PN: [MessageHandler(filters.TEXT & ~filters.COMMAND, in_pn),
             CallbackQueryHandler(cb_handler, pattern="^cancel$")],
        VL: [CallbackQueryHandler(select_vl, pattern="^lang_"),
             CallbackQueryHandler(cb_handler, pattern="^cancel$")],
        SO: [CallbackQueryHandler(cb_handler, pattern="^(o|t|cancel)$")],
        TL: [CallbackQueryHandler(select_tl, pattern="^lang_"),
             CallbackQueryHandler(cb_handler, pattern="^cancel$")],
        VF: [MessageHandler((filters.VIDEO | filters.Document.VIDEO) & ~filters.COMMAND, in_vf),
             CallbackQueryHandler(cb_handler, pattern="^cancel$")]
    },
    fallbacks=[CommandHandler("cancel", cb_handler)]
)

customize_conv = ConversationHandler(
    entry_points=[CallbackQueryHandler(customize_cb_handler, pattern="^(c_|customize)$")],
    states={
        C_MAIN: [CallbackQueryHandler(customize_cb_handler, pattern="^(c_font|c_size|c_text|c_bg|c_time|c_edit|c_save|c_cancel|c_back)$")],
        C_FONT: [CallbackQueryHandler(customize_cb_handler, pattern="^font_")],
        C_SIZE: [CallbackQueryHandler(customize_cb_handler, pattern="^size_")],
        C_TEXT: [CallbackQueryHandler(customize_cb_handler, pattern="^text_"), MessageHandler(filters.TEXT, c_set_textcolor)],
        C_BG: [CallbackQueryHandler(customize_cb_handler, pattern="^bg_")],
        C_TIME: [CallbackQueryHandler(customize_cb_handler, pattern="^time_")],
        C_EDIT: [MessageHandler(filters.TEXT & ~filters.COMMAND, c_set_edit)]
    },
    fallbacks=[CommandHandler("cancel", c_cancel)]
)

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("history", history_command))
app.add_handler(main_conv)
app.add_error_handler(error_handler)
app.add_handler(CallbackQueryHandler(final_cb_handler, pattern="^(download|main)$"))
app.add_handler(CallbackQueryHandler(history_download_handler, pattern="^history_download_"))
app.add_handler(customize_conv)
app.run_polling()
