document.addEventListener("DOMContentLoaded", () => {
    
    // State variables
    let currentMode = "detection";
    let confThresh = 0.15;
    let soundEnabled = true;
    let alarmActive = false;
    let webcamStream = null;
    let vehicleChart = null;

    // DOM Elements
    const liveClock = document.getElementById("liveClock");
    const alertBanner = document.getElementById("alertBanner");
    const alertIcon = document.getElementById("alertIcon");
    const alertBadge = document.getElementById("alertBadge");
    const alertTitle = document.getElementById("alertTitle");
    const alertDesc = document.getElementById("alertDesc");
    const btnSoundToggle = document.getElementById("btnSoundToggle");
    const alarmAudio = document.getElementById("alarmAudio");

    const confSlider = document.getElementById("confSlider");
    const confVal = document.getElementById("confVal");
    const segBtns = document.querySelectorAll(".seg-btn");

    const fileInput = document.getElementById("fileInput");
    const dropzone = document.getElementById("dropzone");
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    const webcamVideo = document.getElementById("webcamVideo");
    const webcamCanvas = document.getElementById("webcamCanvas");
    const webcamOverlay = document.getElementById("webcamOverlay");
    const btnStartCam = document.getElementById("btnStartCam");
    const btnCaptureCam = document.getElementById("btnCaptureCam");

    const samplesGrid = document.getElementById("samplesGrid");
    const loadingOverlay = document.getElementById("loadingOverlay");
    const placeholderView = document.getElementById("placeholderView");
    const resultImage = document.getElementById("resultImage");
    const processTimeTag = document.getElementById("processTimeTag");

    const valTotalVehicles = document.getElementById("valTotalVehicles");
    const valPcuScore = document.getElementById("valPcuScore");
    const cntXeMay = document.getElementById("cntXeMay");
    const cntOTo = document.getElementById("cntOTo");
    const cntXeKhach = document.getElementById("cntXeKhach");
    const cntXeTai = document.getElementById("cntXeTai");
    const valDensity = document.getElementById("valDensity");
    const valFlow = document.getElementById("valFlow");

    const historyTableBody = document.getElementById("historyTableBody");
    const btnClearHistory = document.getElementById("btnClearHistory");

    // Live Clock Update
    function updateClock() {
        const now = new Date();
        liveClock.textContent = now.toLocaleTimeString("vi-VN");
    }
    setInterval(updateClock, 1000);
    updateClock();

    // Init Chart
    function initChart() {
        const ctx = document.getElementById("vehicleChart").getContext("2d");
        vehicleChart = new Chart(ctx, {
            type: "doughnut",
            data: {
                labels: ["Xe máy", "Ô tô con", "Xe khách", "Xe tải"],
                datasets: [{
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        "#ff901e", // Xe máy
                        "#00d7ff", // Ô tô
                        "#e22b8a", // Xe khách
                        "#4370ff"  // Xe tải
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: { color: "#94a3b8", font: { family: "Plus Jakarta Sans", size: 12 } }
                    }
                },
                cutout: "70%"
            }
        });
    }
    initChart();

    // Fetch Preset Samples
    async function loadSamples() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            samplesGrid.innerHTML = "";

            if (data.samples && data.samples.length > 0) {
                data.samples.forEach(s => {
                    const div = document.createElement("div");
                    div.className = "sample-item";
                    div.innerHTML = `
                        <img src="${s.url}" alt="${s.title}">
                        <div class="sample-title">${s.title}</div>
                    `;
                    div.addEventListener("click", () => {
                        processImageUrl(s.url);
                    });
                    samplesGrid.appendChild(div);
                });
            } else {
                samplesGrid.innerHTML = "<p style='color:#94a3b8; font-size:0.8rem;'>Chưa có ảnh mẫu</p>";
            }
        } catch (e) {
            console.error("Failed to load samples:", e);
        }
    }
    loadSamples();

    // Mode Switcher
    segBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            segBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentMode = btn.dataset.mode;
        });
    });

    // Slider
    confSlider.addEventListener("input", (e) => {
        confThresh = e.target.value / 100;
        confVal.textContent = e.target.value + "%";
    });

    // Sound toggle
    btnSoundToggle.addEventListener("click", () => {
        soundEnabled = !soundEnabled;
        if (soundEnabled) {
            btnSoundToggle.classList.add("active");
            btnSoundToggle.innerHTML = '<i class="fa-solid fa-volume-high"></i> Âm thanh';
        } else {
            btnSoundToggle.classList.remove("active");
            btnSoundToggle.innerHTML = '<i class="fa-solid fa-volume-xmark"></i> Tắt âm';
            alarmAudio.pause();
            alarmActive = false;
        }
    });

    // DOM Elements for Video
    const videoDropzone = document.getElementById("videoDropzone");
    const videoFileInput = document.getElementById("videoFileInput");
    const btnPlaySampleVideo = document.getElementById("btnPlaySampleVideo");
    const videoStream = document.getElementById("videoStream");

    // Tabs
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            btn.classList.add("active");
            const target = btn.dataset.tab;
            if (target === "upload") {
                document.getElementById("tabUpload").classList.add("active");
            } else if (target === "video") {
                document.getElementById("tabVideo").classList.add("active");
            } else {
                document.getElementById("tabWebcam").classList.add("active");
            }
        });
    });

    // Video Dropzone & Upload
    if (videoDropzone && videoFileInput) {
        videoDropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            videoDropzone.style.borderColor = "#4facfe";
        });
        videoDropzone.addEventListener("dragleave", () => {
            videoDropzone.style.borderColor = "rgba(79, 172, 254, 0.3)";
        });
        videoDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            videoDropzone.style.borderColor = "rgba(79, 172, 254, 0.3)";
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                uploadVideoFile(e.dataTransfer.files[0]);
            }
        });
        videoFileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                uploadVideoFile(e.target.files[0]);
            }
        });
    }

    if (btnPlaySampleVideo) {
        btnPlaySampleVideo.addEventListener("click", () => {
            playVideoStream("sample_traffic.mp4");
        });
    }

    async function uploadVideoFile(file) {
        showLoading(true);
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/upload_video", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            showLoading(false);

            if (data.success) {
                playVideoStream(data.filename);
            } else {
                alert("Lỗi tải video: " + data.error);
            }
        } catch (e) {
            showLoading(false);
            alert("Lỗi kết nối upload video: " + e.message);
        }
    }

    let videoPollInterval = null;

    function playVideoStream(filename) {
        if (videoPollInterval) clearInterval(videoPollInterval);

        placeholderView.style.display = "none";
        resultImage.style.display = "none";
        videoStream.style.display = "block";
        videoStream.src = `/video_feed/${filename}?conf=${confThresh}&t=${Date.now()}`;
        processTimeTag.innerHTML = `<i class="fa-solid fa-video"></i> Realtime Stream`;

        // Start polling video frame metrics
        let lastLoggedTime = 0;
        videoPollInterval = setInterval(async () => {
            try {
                const res = await fetch(`/api/video_metrics/${filename}`);
                if (!res.ok) return;
                const data = await res.json();
                if (data.success && data.metrics) {
                    const m = data.metrics;
                    valTotalVehicles.textContent = m.total_vehicles;
                    valPcuScore.textContent = m.pcu_score;
                    valDensity.innerHTML = `${m.density != null ? m.density : 0} <small>xe/km</small>`;
                    valFlow.textContent = (m.flow != null) ? Math.round(m.flow) : "\u2014";

                    const c = m.counts;
                    cntXeMay.textContent = c.xe_may;
                    cntOTo.textContent = c.o_to_con;
                    cntXeKhach.textContent = c.xe_khach;
                    cntXeTai.textContent = c.xe_tai;

                    // Update Chart
                    vehicleChart.data.datasets[0].data = [c.xe_may, c.o_to_con, c.xe_khach, c.xe_tai];
                    vehicleChart.update();

                    // Stream đã kết thúc (đã tua hết số lần cho phép) -> dừng polling
                    if (m.stream_ended) {
                        clearInterval(videoPollInterval);
                        videoPollInterval = null;
                        alertBadge.textContent = "ĐÃ KẾT THÚC";
                        alertBadge.style.backgroundColor = "#64748b";
                        alertBadge.style.color = "#fff";
                        alertTitle.textContent = "Luồng video đã kết thúc";
                        alertDesc.textContent = m.stream_message ||
                            "Video đã được phát hết. Vui lòng tải video mới hoặc dùng lại video mẫu.";
                        alertIcon.innerHTML = '<i class="fa-solid fa-circle-info" style="color:#94a3b8"></i>';
                        syncCongestionAlarm(0);
                        return;
                    }

                    // Update Alert Banner
                    const cg = m.congestion;
                    alertBanner.className = "alert-banner status-" + cg.badge;
                    alertBadge.textContent = cg.name.toUpperCase();
                    alertBadge.style.backgroundColor = cg.color;
                    alertBadge.style.color = "#000";

                    alertTitle.textContent = `Mức độ Giao thông (Video): ${cg.name}`;
                    alertDesc.textContent = cg.description;

                    if (cg.level === 0) alertIcon.innerHTML = '<i class="fa-solid fa-circle-minus" style="color:#94a3b8"></i>';
                    else if (cg.level === 1) alertIcon.innerHTML = '<i class="fa-solid fa-circle-check" style="color:#10b981"></i>';
                    else if (cg.level === 2) alertIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation" style="color:#f59e0b"></i>';
                    else if (cg.level >= 3) alertIcon.innerHTML = '<i class="fa-solid fa-bell" style="color:#ef4444"></i>';

                    syncCongestionAlarm(cg.level);

                    // Periodically add to log table (every 3 seconds)
                    const nowSec = Math.floor(Date.now() / 3000);
                    if (nowSec > lastLoggedTime) {
                        lastLoggedTime = nowSec;
                        addHistoryRow({
                            total_vehicles: m.total_vehicles,
                            pcu_score: m.pcu_score,
                            counts: c,
                            congestion: cg,
                            process_time_ms: 30
                        });
                    }
                }
            } catch (err) {
                // Ignore poll errors
            }
        }, 400);
    }

    // Dropzone logic
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "#00f2fe";
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "rgba(0, 242, 254, 0.3)";
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "rgba(0, 242, 254, 0.3)";
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            uploadFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            uploadFile(e.target.files[0]);
        }
    });

    // Webcam handlers
    btnStartCam.addEventListener("click", async () => {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
            webcamVideo.srcObject = webcamStream;
            webcamOverlay.style.display = "none";
            btnCaptureCam.disabled = false;
            btnStartCam.innerHTML = '<i class="fa-solid fa-check"></i> Camera Sẵn Sàng';
        } catch (err) {
            alert("Không thể truy cập camera: " + err.message);
        }
    });

    btnCaptureCam.addEventListener("click", () => {
        if (!webcamStream) return;
        webcamCanvas.width = webcamVideo.videoWidth;
        webcamCanvas.height = webcamVideo.videoHeight;
        const ctx = webcamCanvas.getContext("2d");
        ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);
        const b64 = webcamCanvas.toDataURL("image/jpeg", 0.9);
        processBase64Image(b64);
    });

    // API calls
    async function uploadFile(file) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("mode", currentMode);
        formData.append("conf", confThresh);

        sendApiRequest(formData, false);
    }

    async function processImageUrl(url) {
        try {
            showLoading(true);
            const res = await fetch(url);
            const blob = await res.blob();
            const formData = new FormData();
            formData.append("file", blob, "sample.jpg");
            formData.append("mode", currentMode);
            formData.append("conf", confThresh);

            sendApiRequest(formData, false);
        } catch (e) {
            showLoading(false);
            alert("Lỗi khi tải mẫu ảnh: " + e.message);
        }
    }

    async function processBase64Image(b64) {
        const payload = {
            image_base64: b64,
            mode: currentMode,
            conf: confThresh
        };

        sendApiRequest(payload, true);
    }

    async function sendApiRequest(data, isJson) {
        showLoading(true);
        try {
            const options = {
                method: "POST"
            };
            if (isJson) {
                options.headers = { "Content-Type": "application/json" };
                options.body = JSON.stringify(data);
            } else {
                options.body = data;
            }

            const res = await fetch("/api/predict", options);
            const result = await res.json();
            showLoading(false);

            if (result.success) {
                renderResults(result);
            } else {
                alert("Lỗi phân tích: " + (result.error || "Unknown error"));
            }
        } catch (err) {
            showLoading(false);
            alert("Lỗi kết nối server: " + err.message);
        }
    }

    function showLoading(active) {
        if (active) {
            loadingOverlay.classList.add("active");
        } else {
            loadingOverlay.classList.remove("active");
        }
    }

    // Start the alarm once when congestion begins; stop it when traffic clears.
    function syncCongestionAlarm(level) {
        if (level >= 3 && soundEnabled) {
            if (!alarmActive) {
                alarmActive = true;
                alarmAudio.currentTime = 0;
                alarmAudio.play().catch(() => {
                    // Playback can be blocked until the browser receives user interaction.
                    alarmActive = false;
                });
            }
        } else {
            alarmAudio.pause();
            alarmAudio.currentTime = 0;
            alarmActive = false;
        }
    }

    // Render results to UI
    function renderResults(data) {
        if (videoPollInterval) {
            clearInterval(videoPollInterval);
            videoPollInterval = null;
        }

        // Display image
        placeholderView.style.display = "none";
        videoStream.style.display = "none";
        videoStream.src = "";
        resultImage.style.display = "block";
        resultImage.src = data.annotated_image;

        // Process time
        processTimeTag.innerHTML = `<i class="fa-solid fa-bolt-lightning"></i> ${data.process_time_ms} ms`;

        // Stat counters
        valTotalVehicles.textContent = data.total_vehicles;
        valPcuScore.textContent = data.pcu_score;
        valDensity.innerHTML = `${data.density != null ? data.density : 0} <small>xe/km</small>`;
        valFlow.textContent = (data.flow != null) ? data.flow : "\u2014";

        const c = data.counts;
        cntXeMay.textContent = c.xe_may;
        cntOTo.textContent = c.o_to_con;
        cntXeKhach.textContent = c.xe_khach;
        cntXeTai.textContent = c.xe_tai;

        // Update Chart
        vehicleChart.data.datasets[0].data = [c.xe_may, c.o_to_con, c.xe_khach, c.xe_tai];
        vehicleChart.update();

        // Congestion Alert Banner Update
        const cg = data.congestion;
        alertBanner.className = "alert-banner status-" + cg.badge;
        alertBadge.textContent = cg.name.toUpperCase();
        alertBadge.style.backgroundColor = cg.color;
        alertBadge.style.color = "#000";

        alertTitle.textContent = `Mức độ Giao thông: ${cg.name}`;
        alertDesc.textContent = cg.description;

        // Set Icon
        if (cg.level === 0) alertIcon.innerHTML = '<i class="fa-solid fa-circle-minus" style="color:#94a3b8"></i>';
        else if (cg.level === 1) alertIcon.innerHTML = '<i class="fa-solid fa-circle-check" style="color:#10b981"></i>';
        else if (cg.level === 2) alertIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation" style="color:#f59e0b"></i>';
        else if (cg.level >= 3) alertIcon.innerHTML = '<i class="fa-solid fa-bell" style="color:#ef4444"></i>';

        syncCongestionAlarm(cg.level);

        // Add to history log
        addHistoryRow(data);
    }

    function addHistoryRow(data) {
        const emptyRow = historyTableBody.querySelector(".empty-row");
        if (emptyRow) emptyRow.remove();

        const tr = document.createElement("tr");
        const timeStr = new Date().toLocaleTimeString("vi-VN");
        const cg = data.congestion;

        tr.innerHTML = `
            <td><strong>${timeStr}</strong></td>
            <td><strong>${data.total_vehicles}</strong> xe (PCU: ${data.pcu_score})</td>
            <td>🏍️ ${data.counts.xe_may} | 🚗 ${data.counts.o_to_con} | 🚌 ${data.counts.xe_khach} | 🚚 ${data.counts.xe_tai}</td>
            <td><span class="status-pill" style="background:${cg.bg_color}; color:${cg.color}">${cg.name}</span></td>
            <td>${data.process_time_ms} ms</td>
        `;

        historyTableBody.insertBefore(tr, historyTableBody.firstChild);
    }

    btnClearHistory.addEventListener("click", () => {
        historyTableBody.innerHTML = '<tr class="empty-row"><td colspan="5">Chưa có nhật ký kiểm thử. Hãy tải ảnh lên để bắt đầu.</td></tr>';
    });

});
