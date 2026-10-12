import time, os, shutil
from playwright.sync_api import sync_playwright

os.makedirs("test_artifacts", exist_ok=True)
os.makedirs("test_artifacts/videos", exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        viewport={"width": 1500, "height": 920},
        record_video_dir="test_artifacts/videos"
    )
    page = context.new_page()

    console_errors = []
    page.on("console", lambda msg: console_errors.append(f"[{msg.type}] {msg.text}") if msg.type in ["error"] else None)
    page.on("pageerror", lambda exc: console_errors.append(f"[EXCEPTION] {exc}"))

    print("1. Opening /login...")
    page.goto("http://127.0.0.1:5000/login")
    page.screenshot(path="test_artifacts/01_login_page.png")

    print("2. Logging in as admin...")
    page.fill("#username", "admin")
    page.fill("#password", "admin@123")
    page.click("button[type='submit']")
    page.wait_for_load_state("networkidle")
    time.sleep(2.5)

    print("3. Capturing Overview Tab (with enlarged camera cards and live network flow chart)...")
    page.screenshot(path="test_artifacts/02_tab_overview.png")

    tabs = [
        ("live", "Live Monitoring"),
        ("analytics", "Analytics"),
        ("forecast", "Forecast"),
        ("sources", "Video Sources"),
        ("logs", "Audit Logs"),
        ("settings", "Settings")
    ]

    for tab_id, name in tabs:
        print(f"4. Switching to tab [{tab_id}]...")
        selector = f".nav-tab[data-tab='{tab_id}']"
        page.wait_for_selector(selector)
        page.click(selector)
        # Đợi để dữ liệu tải và Chart.js render xong
        time.sleep(2.5)
        
        panel_selector = f"#tabPanel-{tab_id}"
        display_val = page.evaluate(f"() => window.getComputedStyle(document.querySelector('{panel_selector}')).display")
        class_list = page.evaluate(f"() => Array.from(document.querySelector('{panel_selector}').classList)")
        
        print(f"   -> Result for {tab_id}: display={display_val}, active={'active' in class_list}")
        page.screenshot(path=f"test_artifacts/03_tab_{tab_id}.png")

    # Quay lại tab Phân tích và test đổi camera select
    print("5. Testing analytics camera select switch...")
    page.click(".nav-tab[data-tab='analytics']")
    time.sleep(1)
    page.select_option("#analyticsCameraSelect", "CAM-01")
    time.sleep(2.5)
    page.screenshot(path="test_artifacts/04_analytics_cam01.png")

    # Kiểm tra tab Dự báo đổi horizon và model
    print("6. Testing forecast model switch...")
    page.click(".nav-tab[data-tab='forecast']")
    time.sleep(1)
    page.select_option("#forecastCamSelect", "CAM-01")
    time.sleep(2.5)
    page.screenshot(path="test_artifacts/05_forecast_cam01.png")

    print("\n--- BROWSER CONSOLE ERRORS ---")
    for err in console_errors:
        print(err)
    if not console_errors:
        print("PERFECT: ZERO console errors detected!")
    print("------------------------------\n")

    video_path = page.video.path()
    page.close()
    context.close()
    browser.close()

    if video_path and os.path.exists(video_path):
        target_video = "static/test_demo_tabs.webm"
        shutil.copy(video_path, target_video)
        print(f"Recorded demo video copied to {target_video}")

print("ALL TESTS COMPLETE!")
