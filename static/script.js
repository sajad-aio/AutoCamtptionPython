document.addEventListener("DOMContentLoaded", function () {
  // ********** بخش مربوط به پاپ‌آپ ایجاد پروژه و لودینگ **********
  const modal = document.getElementById("createProjectModal");
  const openBtn = document.querySelector(".create-project-btn");
  const closeBtn = document.querySelector(".btn-secondary");
  const loadingOverlay = document.getElementById("loadingOverlay");

  // تابع باز کردن پاپ‌آپ
  function openCreateProjectModal() {
    console.log("openCreateProjectModal triggered");
    if(modal){
      modal.style.display = "flex";
      setTimeout(() => {
        modal.style.opacity = "1";
      }, 10);
    }
  }

  // تابع بستن پاپ‌آپ
  function closeCreateProjectModal() {
    console.log("closeCreateProjectModal triggered");
    if(modal){
      modal.style.opacity = "0";
      setTimeout(() => {
        modal.style.display = "none";
      }, 300);
    }
  }

  // تابع تغییر نمایش زبان ترجمه (برای فرم ایجاد پروژه)
  function toggleTranslateLang() {
    var option = document.getElementById("subtitle_option").value;
    var translateDiv = document.getElementById("translateDiv");
    if (option === "translated" || option === "bilingual") {
      translateDiv.style.display = "block";
    } else {
      translateDiv.style.display = "none";
    }
  }

  // تابع نمایش لودینگ هنگام ارسال فرم
  function showLoadingOverlay() {
    if(loadingOverlay){
      loadingOverlay.style.display = "flex";
    }
  }

  // ********** بخش مربوط به صفحه ویرایش زیرنویس **********
  // تابع به‌روز‌رسانی استایل overlay زیرنویس
  function updateSubtitleStyle() {
    var subtitleText = document.getElementById('subtitleText');
    var fontFamily = document.getElementById('fontFamily').value;
    var fontSize = document.getElementById('fontSize').value + 'px';
    var bgColor = document.getElementById('bgColor').value;
    var textColor = document.getElementById('textColor').value;
    
    // اعمال استایل‌ها به overlay
    subtitleText.style.fontFamily = fontFamily;
    subtitleText.style.fontSize = fontSize;
    // استفاده از setProperty با !important برای اطمینان از تغییر پس‌زمینه
    subtitleText.style.setProperty("background-color", bgColor, "important");
    subtitleText.style.color = textColor;
    subtitleText.style.padding = '0.5rem';
    
    // به‌روز‌رسانی فیلدهای مخفی برای ارسال به سرور
    document.getElementById('hiddenFontFamily').value = fontFamily;
    document.getElementById('hiddenFontSize').value = document.getElementById('fontSize').value;
    document.getElementById('hiddenBgColor').value = bgColor;
    document.getElementById('hiddenTextColor').value = textColor;
  }

  // تابع تغییر زبان زیرنویس (برای ویرایش صفحه)
  function changeSubtitleLanguage() {
    var lang = document.getElementById('subtitle-lang').value;
    var projectId = "{{ project.id }}";
    fetch("{{ url_for('translate_subtitles') }}?project_id=" + projectId + "&target_lang=" + lang)
      .then(response => response.json())
      .then(data => {
        if(data.translated_text) {
          document.getElementById('subtitleText').innerText = data.translated_text;
        }
      })
      .catch(error => {
        console.error('Translation error:', error);
      });
  }



  // ********** افزودن رویدادها **********
  if (openBtn) {
    openBtn.addEventListener("click", openCreateProjectModal);
  }
  if (closeBtn) {
    closeBtn.addEventListener("click", closeCreateProjectModal);
  }
  if (modal) {
    modal.addEventListener("click", function (e) {
      if (e.target === modal) {
        closeCreateProjectModal();
      }
    });
  }

  function updateSubtitleStyle() {
    var subtitleText = document.getElementById('subtitleText');
    var fontFamily = document.getElementById('fontFamily').value;
    var fontSize = document.getElementById('fontSize').value + 'px';
    var bgColor = document.getElementById('bgColor').value;
    var textColor = document.getElementById('textColor').value;
    var borderRadius = document.getElementById('borderRadius').value;
    
    subtitleText.style.fontFamily = fontFamily;
    subtitleText.style.fontSize = fontSize;
    subtitleText.style.backgroundColor = bgColor;
    subtitleText.style.color = textColor;
    // برای نمونه، می‌توانید با CSS گردی گوشه‌ها را نیز تنظیم کنید
    subtitleText.style.borderRadius = borderRadius + "px";
    
    document.getElementById('hiddenFontFamily').value = fontFamily;
    document.getElementById('hiddenFontSize').value = document.getElementById('fontSize').value;
    document.getElementById('hiddenBgColor').value = bgColor;
    document.getElementById('hiddenTextColor').value = textColor;
    document.getElementById('hiddenBorderRadius').value = borderRadius;
  }
  


  // قرار دادن توابع به صورت global تا در HTML قابل فراخوانی باشند
  window.openCreateProjectModal = openCreateProjectModal;
  window.closeCreateProjectModal = closeCreateProjectModal;
  window.toggleTranslateLang = toggleTranslateLang;
  window.showLoadingOverlay = showLoadingOverlay;
  window.updateSubtitleStyle = updateSubtitleStyle;
  window.changeSubtitleLanguage = changeSubtitleLanguage;
});