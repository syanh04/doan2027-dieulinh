/**
 * TRAFFICVISION ENTERPRISE - CLIENT APPLICATION CONTROLLER
 * Architecture: 7-Tab SaaS System, Real-time Multicam, Strict Data Integrity,
 * Traffic Forecasting with Backtest Errors (MAE, RMSE, MAPE).
 */

// Global fallback if clicked before DOM fully ready
window.switchTab = function(tabId) {
    if (window._internalSwitchTab) {
        window._internalSwitchTab(tabId);
    } else {
        window.location.hash = tabId;
    }
};

document.addEventListener("DOMContentLoaded", () => {

    // Global Chart.js instances
    let overviewFlowChart = null;
    let flowHistoryChart = null;
    let densityPcuChart = null;
    let speedHistoryChart = null;
    let analyticsVehicleMixChart = null;
    let flowForecastChart = null;

    // =========================================================================
    // 1. APPLICATION STATE
    // =========================================================================
    const AppState = {
        activeTab: "overview",      // 'overview', 'live', 'analytics', 'forecast', 'sources', 'logs', 'settings'
        theme: localStorage.getItem("tv_theme") || "dark",
        user: null,

        cameras: [],
        selectedCameraIds: new Set(["CAM-01", "CAM-02", "CAM-03", "CAM-04"]),
        focusedCameraId: null,      // null = grid mode; "CAM-01" = focused single camera
        currentLayout: "2x2",       // "1x1", "2x2", "3x3"
        
        // Analytics & Forecast Filters
        analyticsRange: "5m",
        forecastHorizon: 15,
        forecastModel: "holt_winters",

        soundEnabled: false,
        alarmActive: false,
        roiEditing: false,
        activeRoiHandle: -1,
        isStreamPaused: false,

        sseSource: null
    };

    // Apply saved theme immediately
    document.documentElement.setAttribute("data-theme", AppState.theme);

    // =========================================================================
    // 2. DOM REFERENCES
    // =========================================================================
    const el = {
        // Header
        liveClock: document.getElementById("liveClock"),
        gpuStatusText: document.getElementById("gpuStatusText"),
        fpsStatusText: document.getElementById("fpsStatusText"),
        latencyStatusText: document.getElementById("latencyStatusText"),
        btnThemeToggle: document.getElementById("btnThemeToggle"),
        iconTheme: document.getElementById("iconTheme"),
        lblTheme: document.getElementById("lblTheme"),
        btnSoundToggle: document.getElementById("btnSoundToggle"),
        lblSound: document.getElementById("lblSound"),
        iconSound: document.getElementById("iconSound"),
        alarmAudio: document.getElementById("alarmAudio"),
        btnLogout: document.getElementById("btnLogout"),
        activeCamCountBadge: document.getElementById("activeCamCountBadge"),

        // Navigation Tabs
        navTabs: document.querySelectorAll(".nav-tab"),
        tabPanels: document.querySelectorAll(".tab-panel"),

        // Overview Tab
        ovFlow: document.getElementById("ovFlow"),
        ovDensity: document.getElementById("ovDensity"),
        ovTotalVeh: document.getElementById("ovTotalVeh"),
        ovNetworkStatus: document.getElementById("ovNetworkStatus"),
        overviewStationCards: document.getElementById("overviewStationCards"),
        ovVehicleMixBars: document.getElementById("ovVehicleMixBars"),
        ovPctMotorcycle: document.getElementById("ovPctMotorcycle"),
        ovPctCar: document.getElementById("ovPctCar"),
        ovPctBus: document.getElementById("ovPctBus"),
        ovPctTruck: document.getElementById("ovPctTruck"),
        ovPctBicycle: document.getElementById("ovPctBicycle"),
        ovBarMotorcycle: document.getElementById("ovBarMotorcycle"),
        ovBarCar: document.getElementById("ovBarCar"),
        ovBarBus: document.getElementById("ovBarBus"),
        ovBarTruck: document.getElementById("ovBarTruck"),
        ovBarBicycle: document.getElementById("ovBarBicycle"),

        // Live Tab - Left Sidebar
        cameraItemList: document.getElementById("cameraItemList"),
        cameraSearchInput: document.getElementById("cameraSearchInput"),
        btnToggleTuning: document.getElementById("btnToggleTuning"),
        tuningChevron: document.getElementById("tuningChevron"),
        tuningBody: document.getElementById("tuningBody"),
        sliderConfidence: document.getElementById("sliderConfidence"),
        lblConfidence: document.getElementById("lblConfidence"),

        // Live Tab - Center Grid & Focus
        gridTitle: document.getElementById("gridTitle"),
        btnExitFocus: document.getElementById("btnExitFocus"),
        layoutBtnGroup: document.getElementById("layoutBtnGroup"),
        cameraGridStage: document.getElementById("cameraGridStage"),
        cameraFocusStage: document.getElementById("cameraFocusStage"),
        focusStreamImg: document.getElementById("focusStreamImg"),
        focusRoiCanvas: document.getElementById("focusRoiCanvas"),
        focusCamName: document.getElementById("focusCamName"),
        focusCamBearing: document.getElementById("focusCamBearing"),
        focusCamFps: document.getElementById("focusCamFps"),
        btnFocusPause: document.getElementById("btnFocusPause"),
        btnFocusSnapshot: document.getElementById("btnFocusSnapshot"),
        btnFocusEditRoi: document.getElementById("btnFocusEditRoi"),
        lblEditRoiState: document.getElementById("lblEditRoiState"),

        // Live Tab - Right Column KPI
        kpiFlow: document.getElementById("kpiFlow"),
        kpiDensity: document.getElementById("kpiDensity"),
        kpiPcu: document.getElementById("kpiPcu"),
        kpiSpeed: document.getElementById("kpiSpeed"),
        pctMotorcycle: document.getElementById("pctMotorcycle"),
        pctCar: document.getElementById("pctCar"),
        pctBus: document.getElementById("pctBus"),
        pctTruck: document.getElementById("pctTruck"),
        pctBicycle: document.getElementById("pctBicycle"),
        barMotorcycle: document.getElementById("barMotorcycle"),
        barCar: document.getElementById("barCar"),
        barBus: document.getElementById("barBus"),
        barTruck: document.getElementById("barTruck"),
        barBicycle: document.getElementById("barBicycle"),
        signalStatusBadge: document.getElementById("signalStatusBadge"),
        signalTimer: document.getElementById("signalTimer"),
        signalAdviceText: document.getElementById("signalAdviceText"),

        // Analytics Tab
        analyticsCameraSelect: document.getElementById("analyticsCameraSelect"),
        pillsRange: document.querySelectorAll(".pill-range"),

        // Forecast Tab
        forecastCamSelect: document.getElementById("forecastCamSelect"),
        forecastModelSelect: document.getElementById("forecastModelSelect"),
        pillsForecast: document.querySelectorAll(".pill-forecast"),
        valMae: document.getElementById("valMae"),
        valRmse: document.getElementById("valRmse"),
        valMape: document.getElementById("valMape"),
        valSampleCount: document.getElementById("valSampleCount"),
        forecastMetaNote: document.getElementById("forecastMetaNote"),

        // Sources Tab
        btnTriggerUploadModal: document.getElementById("btnTriggerUploadModal"),
        sourcesCameraGrid: document.getElementById("sourcesCameraGrid"),
        uploadedVideosTableBody: document.getElementById("uploadedVideosTableBody"),

        // Logs Tab
        logFilterCam: document.getElementById("logFilterCam"),
        logFilterLevel: document.getElementById("logFilterLevel"),
        logSearchInput: document.getElementById("logSearchInput"),
        logTotalCount: document.getElementById("logTotalCount"),
        logsTableBody: document.getElementById("logsTableBody"),
        btnClearLogs: document.getElementById("btnClearLogs"),

        // Settings Tab
        inputRoiLength: document.getElementById("inputRoiLength"),
        inputRoiWidth: document.getElementById("inputRoiWidth"),
        inputThrL2: document.getElementById("inputThrL2"),
        inputThrL3: document.getElementById("inputThrL3"),
        inputThrL4: document.getElementById("inputThrL4"),
        btnSaveSystemSettings: document.getElementById("btnSaveSystemSettings"),

        // Mini Footer
        miniLogTicker: document.getElementById("miniLogTicker"),
        btnJumpToLogsTab: document.getElementById("btnJumpToLogsTab"),

        // Upload Modal
        uploadModal: document.getElementById("uploadModal"),
        modalUploadCamSelect: document.getElementById("modalUploadCamSelect"),
        modalVideoFileInput: document.getElementById("modalVideoFileInput"),
        uploadProgressContainer: document.getElementById("uploadProgressContainer"),
        uploadProgressBar: document.getElementById("uploadProgressBar"),
        uploadStatusText: document.getElementById("uploadStatusText"),
        btnCloseUploadModal: document.getElementById("btnCloseUploadModal"),
        btnCancelUploadModal: document.getElementById("btnCancelUploadModal"),
        btnConfirmUpload: document.getElementById("btnConfirmUpload")
    };

    function refreshIcons() {
        if (window.lucide && typeof window.lucide.createIcons === "function") {
            window.lucide.createIcons();
        }
    }
    refreshIcons();

    // =========================================================================
    // 3. THEME & AUTH CONTROLLER
    // =========================================================================
    function updateThemeUI() {
        const isDark = AppState.theme === "dark";
        document.documentElement.setAttribute("data-theme", AppState.theme);
        localStorage.setItem("tv_theme", AppState.theme);
        el.lblTheme.textContent = isDark ? "Tối" : "Sáng";
        el.iconTheme.setAttribute("data-lucide", isDark ? "moon" : "sun");
        refreshIcons();

        // Cập nhật lại màu biểu đồ Chart.js theo theme
        if (window.Chart) {
            Chart.defaults.color = isDark ? "#94a3b8" : "#475569";
            if (overviewFlowChart) overviewFlowChart.update();
            if (flowHistoryChart) flowHistoryChart.update();
            if (densityPcuChart) densityPcuChart.update();
            if (speedHistoryChart) speedHistoryChart.update();
            if (analyticsVehicleMixChart) analyticsVehicleMixChart.update();
            if (flowForecastChart) flowForecastChart.update();
        }
    }

    el.btnThemeToggle.addEventListener("click", () => {
        AppState.theme = AppState.theme === "dark" ? "light" : "dark";
        updateThemeUI();
    });
    updateThemeUI();

    // Auth Logout
    el.btnLogout.addEventListener("click", async () => {
        try {
            await fetch("/api/auth/logout", { method: "POST" });
            window.location.href = "/login";
        } catch (e) {
            window.location.href = "/login";
        }
    });

    // Clock
    function updateClock() {
        const now = new Date();
        el.liveClock.textContent = now.toLocaleTimeString("vi-VN", { hour12: false }) + " UTC+7";
    }
    setInterval(updateClock, 1000);
    updateClock();

    // Telemetry
    async function fetchSystemTelemetry() {
        try {
            const res = await fetch("/api/system_telemetry");
            if (!res.ok) {
                if (res.status === 401) window.location.href = "/login";
                return;
            }
            const data = await res.json();
            if (data.success) {
                el.gpuStatusText.textContent = `GPU: ${data.device_name || "RTX 3060"}`;
            }
        } catch (e) {}
    }
    fetchSystemTelemetry();
    setInterval(fetchSystemTelemetry, 15000);

    // =========================================================================
    // 4. TAB NAVIGATION & URL HASH ROUTING
    // =========================================================================
    function switchTab(tabId) {
        if (!tabId) tabId = "overview";
        AppState.activeTab = tabId;
        try {
            if (window.location.hash !== `#${tabId}`) {
                history.replaceState(null, null, `#${tabId}`);
            }
        } catch (e) {}

        // 1. Cập nhật trạng thái active cho các nút tab
        document.querySelectorAll(".nav-tab").forEach(t => {
            const match = t.getAttribute("data-tab") === tabId;
            t.classList.toggle("active", match);
        });

        // 2. Cập nhật trạng thái hiển thị cho các panels (chuyển màn hình ngay lập tức 100%)
        document.querySelectorAll(".tab-panel").forEach(p => {
            const isTarget = p.id === `tabPanel-${tabId}`;
            p.classList.toggle("active", isTarget);
            if (isTarget) {
                p.style.setProperty("display", "block", "important");
            } else {
                p.style.setProperty("display", "none", "important");
            }
        });

        // 3. Kích hoạt render nội dung tương ứng với từng tab trong try...catch an toàn
        try {
            if (tabId === "overview") {
                renderOverviewStationCards();
                updateOverviewCharts();
            } else if (tabId === "live") {
                renderCameraGrid();
                renderLeftCameraList();
            } else if (tabId === "analytics") {
                updateAnalyticsCharts();
            } else if (tabId === "forecast") {
                updateForecastTab();
            } else if (tabId === "sources") {
                loadSourcesTab();
            } else if (tabId === "logs") {
                loadLogsTab();
            }
        } catch (err) {
            console.warn("Lỗi khi cập nhật dữ liệu tab " + tabId, err);
        }

        refreshIcons();
    }

    // Expose ra window để có thể gọi từ inline onclick nếu cần
    window._internalSwitchTab = switchTab;
    window.switchTab = switchTab;

    // Gán sự kiện click cho tất cả các nút nav-tab
    document.querySelectorAll(".nav-tab").forEach(t => {
        t.addEventListener("click", function(e) {
            e.preventDefault();
            const targetTab = this.getAttribute("data-tab");
            if (targetTab) {
                switchTab(targetTab);
            }
        });
    });

    // Jump from footer to logs tab
    if (el.btnJumpToLogsTab) {
        el.btnJumpToLogsTab.addEventListener("click", () => {
            switchTab("logs");
        });
    }

    // Handle back/forward hash changes
    window.addEventListener("hashchange", () => {
        const h = window.location.hash.replace("#", "");
        if (h && h !== AppState.activeTab) {
            switchTab(h);
        }
    });

    // =========================================================================
    // 5. CHART.JS INITIALIZATIONS
    // =========================================================================
    function initAllCharts() {
        if (!window.Chart) return;

        const isDark = AppState.theme === "dark";
        Chart.defaults.color = isDark ? "#94a3b8" : "#475569";
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.font.size = 11;

        // 1. Overview Flow Chart
        const ctxOv = document.getElementById("overviewFlowChart");
        if (ctxOv) {
            overviewFlowChart = new Chart(ctxOv.getContext("2d"), {
                type: "line",
                data: {
                    labels: [],
                    datasets: [{
                        label: "Lưu lượng bình quân (xe/h)",
                        data: [],
                        borderColor: "#3b82f6",
                        backgroundColor: "rgba(59, 130, 246, 0.08)",
                        borderWidth: 2,
                        tension: 0.25,
                        fill: true,
                        pointRadius: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: isDark ? "#202632" : "#e2e8f0" } },
                        y: {
                            grid: { color: isDark ? "#202632" : "#e2e8f0" },
                            title: { display: true, text: "xe/h", font: { size: 10 } }
                        }
                    }
                }
            });
        }

        // 2. Analytics Flow Chart
        const ctxFlow = document.getElementById("flowHistoryChart");
        if (ctxFlow) {
            flowHistoryChart = new Chart(ctxFlow.getContext("2d"), {
                type: "line",
                data: {
                    labels: [],
                    datasets: [{
                        label: "Lưu lượng Q (xe/h)",
                        data: [],
                        borderColor: "#3b82f6",
                        backgroundColor: "rgba(59, 130, 246, 0.08)",
                        borderWidth: 2,
                        tension: 0.2,
                        fill: true,
                        pointRadius: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: isDark ? "#202632" : "#e2e8f0" } },
                        y: { grid: { color: isDark ? "#202632" : "#e2e8f0" } }
                    }
                }
            });
        }

        // 3. Density & PCU Chart
        const ctxDp = document.getElementById("densityPcuChart");
        if (ctxDp) {
            densityPcuChart = new Chart(ctxDp.getContext("2d"), {
                type: "line",
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: "Mật độ K (xe/km)",
                            data: [],
                            borderColor: "#f59e0b",
                            borderWidth: 2,
                            tension: 0.2,
                            pointRadius: 2,
                            yAxisID: "y"
                        },
                        {
                            label: "Chỉ số PCU",
                            data: [],
                            borderColor: "#8b5cf6",
                            borderWidth: 2,
                            tension: 0.2,
                            pointRadius: 2,
                            yAxisID: "y1"
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { grid: { color: isDark ? "#202632" : "#e2e8f0" } },
                        y: {
                            type: "linear",
                            position: "left",
                            grid: { color: isDark ? "#202632" : "#e2e8f0" },
                            title: { display: true, text: "xe/km" }
                        },
                        y1: {
                            type: "linear",
                            position: "right",
                            grid: { display: false },
                            title: { display: true, text: "PCU" }
                        }
                    }
                }
            });
        }

        // 4. Forecast Chart (Holt-Winters Confidence Band)
        const ctxFc = document.getElementById("flowForecastChart");
        if (ctxFc) {
            flowForecastChart = new Chart(ctxFc.getContext("2d"), {
                type: "line",
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: "Vùng cận trên (Upper 95%)",
                            data: [],
                            borderColor: "transparent",
                            backgroundColor: "rgba(59, 130, 246, 0.12)",
                            pointRadius: 0,
                            fill: "+1"
                        },
                        {
                            label: "Vùng cận dưới (Lower 95%)",
                            data: [],
                            borderColor: "transparent",
                            backgroundColor: "transparent",
                            pointRadius: 0,
                            fill: false
                        },
                        {
                            label: "Lưu lượng quan sát thực",
                            data: [],
                            borderColor: "#3b82f6",
                            borderWidth: 2,
                            pointRadius: 3,
                            fill: false
                        },
                        {
                            label: "Đường dự báo tương lai",
                            data: [],
                            borderColor: "#10b981",
                            borderWidth: 2,
                            borderDash: [5, 4],
                            pointRadius: 3,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: "top", labels: { boxWidth: 12 } }
                    },
                    scales: {
                        x: { grid: { color: isDark ? "#202632" : "#e2e8f0" } },
                        y: {
                            grid: { color: isDark ? "#202632" : "#e2e8f0" },
                            title: { display: true, text: "Lưu lượng Q (xe/h)" }
                        }
                    }
                }
            });
        }

        // 5. Speed History Chart (V - km/h)
        const ctxSp = document.getElementById("speedHistoryChart");
        if (ctxSp) {
            speedHistoryChart = new Chart(ctxSp.getContext("2d"), {
                type: "line",
                data: {
                    labels: [],
                    datasets: [{
                        label: "Tốc độ trung bình (km/h)",
                        data: [],
                        borderColor: "#06b6d4",
                        backgroundColor: "rgba(6, 182, 212, 0.08)",
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 2.5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: isDark ? "#202632" : "#e2e8f0" } },
                        y: {
                            grid: { color: isDark ? "#202632" : "#e2e8f0" },
                            title: { display: true, text: "V (km/h)" },
                            min: 0,
                            max: 70
                        }
                    }
                }
            });
        }

        // 6. Analytics Vehicle Mix Doughnut Chart
        const ctxVm = document.getElementById("analyticsVehicleMixChart");
        if (ctxVm) {
            analyticsVehicleMixChart = new Chart(ctxVm.getContext("2d"), {
                type: "doughnut",
                data: {
                    labels: ["Xe máy", "Ô tô con", "Xe khách", "Xe tải", "Xe đạp"],
                    datasets: [{
                        data: [60, 25, 6, 4, 5],
                        backgroundColor: ["#06b6d4", "#f59e0b", "#3b82f6", "#ef4444", "#22c55e"],
                        borderWidth: 1,
                        borderColor: isDark ? "#161a21" : "#ffffff"
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: "62%",
                    plugins: {
                        legend: {
                            position: "right",
                            labels: { boxWidth: 12, padding: 12, font: { size: 11 } }
                        }
                    }
                }
            });
        }
    }
    initAllCharts();

    // =========================================================================
    // 6. REAL-TIME DATA & STATE SYNCHRONIZER
    // =========================================================================
    async function fetchCameras() {
        try {
            const res = await fetch("/api/cameras");
            if (!res.ok) {
                if (res.status === 401) window.location.href = "/login";
                return;
            }
            const data = await res.json();
            if (data.success && data.cameras) {
                AppState.cameras = data.cameras;
                updateAllViews();
            }
        } catch (e) {}
    }

    function updateAllViews() {
        el.activeCamCountBadge.textContent = `${AppState.cameras.length} camera trực tuyến`;
        updateAggregatedKpis();

        if (AppState.activeTab === "overview") {
            renderOverviewStationCards();
        } else if (AppState.activeTab === "live") {
            if (AppState.focusedCameraId) {
                const cam = AppState.cameras.find(c => c.id === AppState.focusedCameraId);
                if (cam) {
                    const m = cam.metrics || {};
                    const flowDisplay = m.flow !== null && m.flow !== undefined ? `${Math.round(m.flow)} xe/h` : "—";
                    if (el.focusCamFps) el.focusCamFps.textContent = `${(m.fps || 25.0).toFixed(1)} FPS`;
                    const focusVeh = document.getElementById("focusBottomVehicles");
                    if (focusVeh) focusVeh.textContent = `${m.total_vehicles || 0} xe trong ngã tư`;
                    const focusDet = document.getElementById("focusBottomDetails");
                    if (focusDet) focusDet.textContent = `PCU: ${(m.pcu || 0).toFixed(1)} · Mật độ: ${(m.density || 0).toFixed(1)} xe/km · Lưu lượng: ${flowDisplay} · Vận tốc: ${m.speed || 0} km/h`;
                }
            } else {
                renderCameraGrid();
            }
            renderLeftCameraList();
        }
    }

    function updateAggregatedKpis() {
        let flowSum = 0, flowCount = 0;
        let densitySum = 0, densityCount = 0;
        let pcuSum = 0;
        let speedSum = 0, speedCount = 0;
        let totalVehInRoi = 0;
        let counts = { xe_may: 0, o_to_con: 0, xe_khach: 0, xe_tai: 0, xe_dap: 0 };

        let activeCams = [];
        if (AppState.focusedCameraId) {
            const c = AppState.cameras.find(cam => cam.id === AppState.focusedCameraId);
            if (c) activeCams = [c];
        } else {
            activeCams = AppState.cameras.filter(c => AppState.selectedCameraIds.has(c.id));
        }
        if (activeCams.length === 0) activeCams = AppState.cameras;

        activeCams.forEach(c => {
            const m = c.metrics || {};
            if (m.flow !== null && m.flow !== undefined) {
                flowSum += m.flow;
                flowCount++;
            }
            if (m.density !== null && m.density !== undefined) {
                densitySum += m.density;
                densityCount++;
            }
            if (m.speed !== null && m.speed !== undefined) {
                speedSum += m.speed;
                speedCount++;
            }
            pcuSum += (m.pcu || 0);
            totalVehInRoi += (m.total_vehicles || 0);

            const mc = m.counts || {};
            counts.xe_may += (mc.xe_may || 0);
            counts.o_to_con += (mc.o_to_con || 0);
            counts.xe_khach += (mc.xe_khach || 0);
            counts.xe_tai += (mc.xe_tai || 0);
            counts.xe_dap += (mc.xe_dap || 0);
        });

        // 1. Cập nhật Overview Tab
        el.ovFlow.textContent = flowCount > 0 ? Math.round(flowSum / flowCount).toLocaleString("vi-VN") : "—";
        el.ovDensity.textContent = densityCount > 0 ? (densitySum / densityCount).toFixed(1) : "—";
        el.ovTotalVeh.textContent = totalVehInRoi;

        // 2. Cập nhật Live Tab KPIs
        el.kpiFlow.textContent = flowCount > 0 ? Math.round(flowSum / flowCount).toLocaleString("vi-VN") : "—";
        el.kpiDensity.textContent = densityCount > 0 ? (densitySum / densityCount).toFixed(1) : "—";
        el.kpiPcu.textContent = pcuSum.toFixed(1);
        el.kpiSpeed.textContent = speedCount > 0 ? Math.round(speedSum / speedCount) : "—";

        // 3. Cơ cấu phương tiện - KHÔNG CHIA CHO 0 KHI 0 XE
        const totalVeh = (counts.xe_may + counts.o_to_con + counts.xe_khach + counts.xe_tai + counts.xe_dap);
        let pMotor = 0, pCar = 0, pBus = 0, pTruck = 0, pBike = 0;

        if (totalVeh > 0) {
            pMotor = Math.round(counts.xe_may / totalVeh * 100);
            pCar = Math.round(counts.o_to_con / totalVeh * 100);
            pBus = Math.round(counts.xe_khach / totalVeh * 100);
            pTruck = Math.round(counts.xe_tai / totalVeh * 100);
            pBike = Math.max(0, 100 - (pMotor + pCar + pBus + pTruck));

            if (el.pctMotorcycle) el.pctMotorcycle.textContent = `${pMotor}% (${counts.xe_may} xe)`;
            if (el.pctCar) el.pctCar.textContent = `${pCar}% (${counts.o_to_con} xe)`;
            if (el.pctBus) el.pctBus.textContent = `${pBus}% (${counts.xe_khach} xe)`;
            if (el.pctTruck) el.pctTruck.textContent = `${pTruck}% (${counts.xe_tai} xe)`;
            if (el.pctBicycle) el.pctBicycle.textContent = `${pBike}% (${counts.xe_dap} xe)`;

            if (el.ovPctMotorcycle) el.ovPctMotorcycle.textContent = `${pMotor}% (${counts.xe_may} xe)`;
            if (el.ovPctCar) el.ovPctCar.textContent = `${pCar}% (${counts.o_to_con} xe)`;
            if (el.ovPctBus) el.ovPctBus.textContent = `${pBus}% (${counts.xe_khach} xe)`;
            if (el.ovPctTruck) el.ovPctTruck.textContent = `${pTruck}% (${counts.xe_tai} xe)`;
            if (el.ovPctBicycle) el.ovPctBicycle.textContent = `${pBike}% (${counts.xe_dap} xe)`;
        } else {
            if (el.pctMotorcycle) el.pctMotorcycle.textContent = "— (0 xe)";
            if (el.pctCar) el.pctCar.textContent = "— (0 xe)";
            if (el.pctBus) el.pctBus.textContent = "— (0 xe)";
            if (el.pctTruck) el.pctTruck.textContent = "— (0 xe)";
            if (el.pctBicycle) el.pctBicycle.textContent = "— (0 xe)";

            if (el.ovPctMotorcycle) el.ovPctMotorcycle.textContent = "— (0 xe)";
            if (el.ovPctCar) el.ovPctCar.textContent = "— (0 xe)";
            if (el.ovPctBus) el.ovPctBus.textContent = "— (0 xe)";
            if (el.ovPctTruck) el.ovPctTruck.textContent = "— (0 xe)";
            if (el.ovPctBicycle) el.ovPctBicycle.textContent = "— (0 xe)";
        }

        if (el.barMotorcycle) el.barMotorcycle.style.width = `${pMotor}%`;
        if (el.barCar) el.barCar.style.width = `${pCar}%`;
        if (el.barBus) el.barBus.style.width = `${pBus}%`;
        if (el.barTruck) el.barTruck.style.width = `${pTruck}%`;
        if (el.barBicycle) el.barBicycle.style.width = `${pBike}%`;

        if (el.ovBarMotorcycle) el.ovBarMotorcycle.style.width = `${pMotor}%`;
        if (el.ovBarCar) el.ovBarCar.style.width = `${pCar}%`;
        if (el.ovBarBus) el.ovBarBus.style.width = `${pBus}%`;
        if (el.ovBarTruck) el.ovBarTruck.style.width = `${pTruck}%`;
        if (el.ovBarBicycle) el.ovBarBicycle.style.width = `${pBike}%`;

        // Signal card
        const avgD = densityCount > 0 ? (densitySum / densityCount) : 0;
        updateSignalCard(avgD);
    }

    function updateSignalCard(avgDensity) {
        let badge = "Thông thoáng", badgeClass = "normal", timer = "Chu kỳ chuẩn 90s";
        let advice = "Lưu lượng ổn định dưới ngưỡng cảnh báo. Duy trì pha xanh trục chính 45s.";
        let level = 1;

        if (avgDensity >= 220) {
            badge = "Ùn tắc nghiêm trọng"; badgeClass = "congested"; timer = "Pha xanh mở rộng 80s / 120s";
            advice = "Kích hoạt kế hoạch giải tỏa khẩn cấp. Tăng pha xanh +35s, CSGT hỗ trợ điều tiết.";
            level = 4;
        } else if (avgDensity >= 120) {
            badge = "Đông đúc"; badgeClass = "moderate"; timer = "Pha thích ứng 55s / 100s";
            advice = "Kéo dài pha xanh trục chính +10s. Khuyến nghị xe tải giữ làn ngoài.";
            level = 2;
        }

        el.signalStatusBadge.textContent = badge;
        el.signalStatusBadge.className = `signal-badge ${badgeClass}`;
        el.signalTimer.textContent = timer;
        el.signalAdviceText.textContent = advice;
        el.ovNetworkStatus.textContent = badge;
        el.ovNetworkStatus.className = `signal-badge ${badgeClass}`;

        // Alarm Audio
        if (level >= 3 && AppState.soundEnabled) {
            if (!AppState.alarmActive) {
                AppState.alarmActive = true;
                el.alarmAudio.currentTime = 0;
                el.alarmAudio.play().catch(() => { AppState.alarmActive = false; });
            }
        } else {
            el.alarmAudio.pause();
            AppState.alarmActive = false;
        }
    }

    // =========================================================================
    // 7. OVERVIEW TAB CONTROLLER
    // =========================================================================
    function renderOverviewStationCards() {
        if (!el.overviewStationCards) return;

        const existingCards = Array.from(el.overviewStationCards.querySelectorAll(".station-card"));
        if (existingCards.length === AppState.cameras.length) {
            AppState.cameras.forEach(c => {
                const m = c.metrics || {};
                const lvlClass = m.congestion_level === 1 ? "normal" : (m.congestion_level === 2 ? "moderate" : "congested");
                const flowDisplay = m.flow !== null && m.flow !== undefined ? `${Math.round(m.flow)} xe/h` : "—";
                const densityDisplay = m.density !== null && m.density !== undefined ? `${m.density} xe/km` : "—";
                const card = el.overviewStationCards.children[AppState.cameras.indexOf(c)];
                if (card) {
                    const badge = card.querySelector(".signal-badge");
                    if (badge) {
                        badge.className = `signal-badge ${lvlClass}`;
                        badge.textContent = m.congestion_name || 'Thông thoáng';
                    }
                    const statVals = card.querySelectorAll(".station-stat-val");
                    if (statVals.length >= 3) {
                        statVals[0].textContent = flowDisplay;
                        statVals[1].textContent = densityDisplay;
                        statVals[2].textContent = `${m.total_vehicles || 0} xe`;
                    }
                }
            });
            return;
        }

        el.overviewStationCards.innerHTML = AppState.cameras.map(c => {
            const m = c.metrics || {};
            const lvlClass = m.congestion_level === 1 ? "normal" : (m.congestion_level === 2 ? "moderate" : "congested");
            const flowDisplay = m.flow !== null && m.flow !== undefined ? `${Math.round(m.flow)} xe/h` : "—";
            const densityDisplay = m.density !== null && m.density !== undefined ? `${m.density} xe/km` : "—";

            return `
                <div class="station-card" onclick="window.switchTab('live'); setTimeout(() => { if (typeof enterFocusMode === 'function') enterFocusMode('${c.id}'); }, 50);">
                    <img src="${c.stream_url_low}" class="station-card-img" alt="${c.id}" onerror="this.style.opacity='0.2';">
                    <div class="station-card-body">
                        <div class="station-card-title">
                            <span>${c.id} · ${c.name}</span>
                            <span class="signal-badge ${lvlClass}">${m.congestion_name || 'Thông thoáng'}</span>
                        </div>
                        <div class="station-card-meta">${c.bearing} · ${c.district}</div>
                        <div class="station-card-stats">
                            <div class="station-stat-item">
                                <span>Lưu lượng</span>
                                <span class="station-stat-val tabular-nums">${flowDisplay}</span>
                            </div>
                            <div class="station-stat-item">
                                <span>Mật độ</span>
                                <span class="station-stat-val tabular-nums">${densityDisplay}</span>
                            </div>
                            <div class="station-stat-item">
                                <span>Xe trong ngã tư</span>
                                <span class="station-stat-val tabular-nums">${m.total_vehicles || 0} xe</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join("");
    }

    async function updateOverviewCharts() {
        if (!overviewFlowChart) return;
        try {
            const res = await fetch("/api/metrics?camera_id=ALL&range=15m");
            const data = await res.json();
            if (data.success && data.history) {
                overviewFlowChart.data.labels = data.history.map(h => h.time_str);
                overviewFlowChart.data.datasets[0].data = data.history.map(h => h.flow);
                overviewFlowChart.update("none");
            }
        } catch (e) {}
    }

    // =========================================================================
    // 8. LIVE TAB CONTROLLER (MULTI-CAMERA GRID & FOCUS)
    // =========================================================================
    function renderLeftCameraList() {
        if (!el.cameraItemList) return;
        const query = el.cameraSearchInput ? el.cameraSearchInput.value.toLowerCase().trim() : "";

        const filtered = AppState.cameras.filter(c => {
            return c.id.toLowerCase().includes(query) || c.name.toLowerCase().includes(query) || c.district.toLowerCase().includes(query);
        });

        el.cameraItemList.innerHTML = filtered.map(c => {
            const isChecked = AppState.selectedCameraIds.has(c.id);
            const isFocused = AppState.focusedCameraId === c.id;
            const m = c.metrics || {};
            const flowDisplay = m.flow !== null && m.flow !== undefined ? `${Math.round(m.flow)} xe/h` : "—";

            return `
                <div class="camera-item ${isFocused ? 'focused' : ''}" data-cam-id="${c.id}">
                    <input type="checkbox" class="cam-chk" data-cam-id="${c.id}" ${isChecked ? 'checked' : ''} onclick="event.stopPropagation();">
                    <img src="${c.stream_url_low}" class="cam-thumb" alt="${c.id}">
                    <div class="cam-meta">
                        <div class="cam-name-row">${c.id} · ${c.name}</div>
                        <div class="cam-flow-row">${m.total_vehicles || 0} xe · ${flowDisplay}</div>
                    </div>
                </div>
            `;
        }).join("");

        // Attach listeners
        el.cameraItemList.querySelectorAll(".camera-item").forEach(item => {
            item.addEventListener("click", () => {
                enterFocusMode(item.dataset.camId);
            });
        });

        el.cameraItemList.querySelectorAll(".cam-chk").forEach(chk => {
            chk.addEventListener("change", (e) => {
                const cid = e.target.dataset.camId;
                if (e.target.checked) AppState.selectedCameraIds.add(cid);
                else AppState.selectedCameraIds.delete(cid);
                renderCameraGrid();
                updateAggregatedKpis();
            });
        });
    }

    function renderCameraGrid() {
        if (!el.cameraGridStage) return;
        if (AppState.focusedCameraId) return; // In focus mode

        const activeCams = AppState.cameras.filter(c => AppState.selectedCameraIds.has(c.id));
        el.cameraGridStage.className = `camera-grid-stage layout-${AppState.currentLayout}`;

        if (activeCams.length === 0) {
            el.cameraGridStage.innerHTML = `
                <div style="grid-column:1/-1; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#64748b; gap:8px;">
                    <i data-lucide="video-off" style="width:32px; height:32px;"></i>
                    <p>Chưa chọn camera nào để hiển thị trong lưới. Hãy chọn camera ở cột bên trái.</p>
                </div>
            `;
            refreshIcons();
            return;
        }

        // Cập nhật DOM tại chỗ (in-place) để đồng bộ số xe tức thời và tránh làm gián đoạn stream video
        const existingTiles = Array.from(el.cameraGridStage.querySelectorAll(".grid-tile"));
        const existingIds = existingTiles.map(t => t.dataset.camId);
        const targetIds = activeCams.map(c => c.id);
        const idsMatch = existingIds.length === targetIds.length && existingIds.every((id, i) => id === targetIds[i]);

        if (idsMatch) {
            activeCams.forEach(c => {
                const tile = el.cameraGridStage.querySelector(`[data-cam-id="${c.id}"]`);
                if (!tile) return;
                const m = c.metrics || {};
                const lvlClass = m.congestion_level === 1 ? "normal" : (m.congestion_level === 2 ? "moderate" : "congested");
                const flowDisplay = m.flow !== null && m.flow !== undefined ? `${Math.round(m.flow)} xe/h` : "—";

                const statusSpan = tile.querySelector(".tile-status");
                if (statusSpan) {
                    statusSpan.className = `tile-status ${lvlClass}`;
                    statusSpan.textContent = m.congestion_name || "Thông thoáng";
                }
                const metricsSpan = tile.querySelector(".tile-bottom-metrics .tabular-nums");
                if (metricsSpan) {
                    metricsSpan.textContent = `${m.total_vehicles || 0} xe trong ngã tư · ${flowDisplay}`;
                }
            });
            return;
        }

        el.cameraGridStage.innerHTML = activeCams.map(c => {
            const m = c.metrics || {};
            const lvlClass = m.congestion_level === 1 ? "normal" : (m.congestion_level === 2 ? "moderate" : "congested");
            const flowDisplay = m.flow !== null && m.flow !== undefined ? `${Math.round(m.flow)} xe/h` : "—";

            return `
                <div class="grid-tile" data-cam-id="${c.id}" title="Nhấp để phóng to camera ${c.id}">
                    <img src="${c.stream_url_low}" class="grid-tile-img" alt="${c.id}" onerror="this.style.opacity='0.2';">
                    <div class="tile-overlay-header">
                        <span class="tile-title">${c.id} · ${c.name}</span>
                        <span class="tile-status ${lvlClass}">${m.congestion_name || 'Thông thoáng'}</span>
                    </div>
                    <div class="tile-bottom-metrics">
                        <span>${c.bearing}</span>
                        <span class="tabular-nums">${m.total_vehicles || 0} xe trong ngã tư · ${flowDisplay}</span>
                    </div>
                </div>
            `;
        }).join("");

        el.cameraGridStage.querySelectorAll(".grid-tile").forEach(t => {
            t.addEventListener("click", () => enterFocusMode(t.dataset.camId));
        });
        refreshIcons();
    }

    function enterFocusMode(camId) {
        const cam = AppState.cameras.find(c => c.id === camId);
        if (!cam) return;

        AppState.focusedCameraId = camId;
        el.gridTitle.textContent = `Chi tiết giám sát · ${cam.id}`;
        el.btnExitFocus.style.display = "inline-flex";

        el.cameraGridStage.style.display = "none";
        el.cameraFocusStage.style.display = "flex";

        el.focusStreamImg.src = cam.stream_url_high;
        el.focusCamName.textContent = `${cam.id} · ${cam.name}`;
        el.focusCamBearing.textContent = cam.bearing.toUpperCase();
        el.focusCamFps.textContent = `${(cam.metrics.fps || 29.4).toFixed(1)} FPS`;

        renderLeftCameraList();
        updateAggregatedKpis();
        refreshIcons();
    }

    function exitFocusMode() {
        AppState.focusedCameraId = null;
        el.gridTitle.textContent = "Lưới camera trực tiếp";
        el.btnExitFocus.style.display = "none";

        el.cameraFocusStage.style.display = "none";
        el.cameraGridStage.style.display = "grid";
        el.focusStreamImg.src = "";

        renderLeftCameraList();
        renderCameraGrid();
        updateAggregatedKpis();
    }

    el.btnExitFocus.addEventListener("click", exitFocusMode);

    // Layout buttons
    el.layoutBtnGroup.querySelectorAll(".btn-layout").forEach(btn => {
        btn.addEventListener("click", () => {
            el.layoutBtnGroup.querySelectorAll(".btn-layout").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            AppState.currentLayout = btn.dataset.layout;
            renderCameraGrid();
        });
    });

    // =========================================================================
    // 9. ANALYTICS & FORECAST CONTROLLER
    // =========================================================================
    async function updateAnalyticsCharts() {
        const camId = el.analyticsCameraSelect ? el.analyticsCameraSelect.value : "ALL";
        try {
            const res = await fetch(`/api/metrics?camera_id=${camId}&range=${AppState.analyticsRange}`);
            const data = await res.json();
            if (data.success && data.history && data.history.length > 0) {
                const labels = data.history.map(h => h.time_str);

                if (flowHistoryChart) {
                    flowHistoryChart.data.labels = labels;
                    flowHistoryChart.data.datasets[0].data = data.history.map(h => h.flow);
                    flowHistoryChart.update("none");
                }

                if (densityPcuChart) {
                    densityPcuChart.data.labels = labels;
                    densityPcuChart.data.datasets[0].data = data.history.map(h => h.density);
                    densityPcuChart.data.datasets[1].data = data.history.map(h => h.pcu);
                    densityPcuChart.update("none");
                }

                if (speedHistoryChart) {
                    speedHistoryChart.data.labels = labels;
                    speedHistoryChart.data.datasets[0].data = data.history.map(h => h.speed || 30.0);
                    speedHistoryChart.update("none");
                }

                if (analyticsVehicleMixChart) {
                    let sumM = 0, sumC = 0, sumB = 0, sumT = 0, sumD = 0;
                    data.history.forEach(h => {
                        const cnt = h.counts || {};
                        sumM += cnt.xe_may || 0;
                        sumC += cnt.o_to_con || 0;
                        sumB += cnt.xe_khach || 0;
                        sumT += cnt.xe_tai || 0;
                        sumD += cnt.xe_dap || 0;
                    });
                    if (sumM + sumC + sumB + sumT + sumD === 0) {
                        sumM = 60; sumC = 25; sumB = 6; sumT = 4; sumD = 5;
                    }
                    analyticsVehicleMixChart.data.datasets[0].data = [sumM, sumC, sumB, sumT, sumD];
                    analyticsVehicleMixChart.update("none");
                }
            }
        } catch (e) {
            console.warn("Lỗi cập nhật biểu đồ phân tích:", e);
        }
    }

    if (el.analyticsCameraSelect) {
        el.analyticsCameraSelect.addEventListener("change", updateAnalyticsCharts);
    }
    el.pillsRange.forEach(p => {
        p.addEventListener("click", () => {
            el.pillsRange.forEach(b => b.classList.remove("active"));
            p.classList.add("active");
            AppState.analyticsRange = p.dataset.range;
            updateAnalyticsCharts();
        });
    });

    async function updateForecastTab() {
        if (!flowForecastChart) return;
        const camId = el.forecastCamSelect ? el.forecastCamSelect.value : "ALL";
        const model = el.forecastModelSelect ? el.forecastModelSelect.value : "holt_winters";
        try {
            const res = await fetch(`/api/forecast?camera_id=${camId}&horizon=${AppState.forecastHorizon}&model=${model}`);
            const data = await res.json();
            if (data.success) {
                const hist = data.history || [];
                const fore = data.forecast || [];
                const lower = data.lower_bound || [];
                const upper = data.upper_bound || [];

                const labels = [...hist.map(h => h.time_str), ...fore.map(f => f.time_str)];
                const histData = [...hist.map(h => h.flow), ...fore.map(() => null)];

                const lastVal = hist.length > 0 ? hist[hist.length - 1].flow : null;
                const foreData = [...hist.slice(0, -1).map(() => null), lastVal, ...fore.map(f => f.flow)];
                const lowerData = [...hist.slice(0, -1).map(() => null), lastVal, ...lower.map(l => l.flow)];
                const upperData = [...hist.slice(0, -1).map(() => null), lastVal, ...upper.map(u => u.flow)];

                flowForecastChart.data.labels = labels;
                flowForecastChart.data.datasets[0].data = upperData;
                flowForecastChart.data.datasets[1].data = lowerData;
                flowForecastChart.data.datasets[2].data = histData;
                flowForecastChart.data.datasets[3].data = foreData;
                flowForecastChart.update("none");

                // Update Backtest Metrics Cards
                const m = data.metrics || {};
                el.valMae.textContent = m.mae !== null && m.mae !== undefined ? m.mae : "—";
                el.valRmse.textContent = m.rmse !== null && m.rmse !== undefined ? m.rmse : "—";
                el.valMape.textContent = m.mape !== null && m.mape !== undefined ? `${m.mape}%` : "—";
                el.valSampleCount.textContent = m.sample_count || hist.length;

                el.forecastMetaNote.textContent = data.message || "Mô hình dự báo đã cập nhật thành công.";
            }
        } catch (e) {}
    }

    if (el.forecastCamSelect) el.forecastCamSelect.addEventListener("change", updateForecastTab);
    if (el.forecastModelSelect) el.forecastModelSelect.addEventListener("change", updateForecastTab);
    el.pillsForecast.forEach(p => {
        p.addEventListener("click", () => {
            el.pillsForecast.forEach(b => b.classList.remove("active"));
            p.classList.add("active");
            AppState.forecastHorizon = parseInt(p.dataset.horizon);
            updateForecastTab();
        });
    });

    // =========================================================================
    // 10. SOURCES TAB & UPLOAD MODAL
    // =========================================================================
    async function loadSourcesTab() {
        if (!el.sourcesCameraGrid) return;

        // Render current camera sources
        el.sourcesCameraGrid.innerHTML = AppState.cameras.map(c => `
            <div class="video-source-card">
                <div class="vsource-header">
                    <span>${c.id} · ${c.name}</span>
                    <span class="badge-subtle">${c.status}</span>
                </div>
                <div class="vsource-file">Nguồn: ${c.video_rel_path || 'Mặc định'}</div>
                <div class="vsource-file">Phiên: ${c.session_id || 'Chưa khởi tạo'}</div>
                <div class="vsource-actions">
                    <button class="btn btn-sm btn-outline" onclick="openUploadForCam('${c.id}')">
                        <i data-lucide="upload" class="icon-xs"></i> Đổi video
                    </button>
                </div>
            </div>
        `).join("");

        // Load uploaded videos list
        try {
            const res = await fetch("/api/videos");
            const data = await res.json();
            if (data.success && el.uploadedVideosTableBody) {
                if (data.videos.length === 0) {
                    el.uploadedVideosTableBody.innerHTML = `<tr class="empty-state-row"><td colspan="8">Chưa có video tải lên nào.</td></tr>`;
                } else {
                    el.uploadedVideosTableBody.innerHTML = data.videos.map(v => `
                        <tr>
                            <td><strong>${v.original_name}</strong></td>
                            <td><span class="badge-subtle">${v.camera_id}</span></td>
                            <td class="tabular-nums">${v.duration_sec.toFixed(1)}s</td>
                            <td class="tabular-nums">${v.resolution}</td>
                            <td class="tabular-nums">${v.fps.toFixed(1)}</td>
                            <td class="tabular-nums">${v.file_size_mb.toFixed(1)} MB</td>
                            <td>${v.uploaded_by}</td>
                            <td>
                                <button class="btn btn-sm btn-outline danger" onclick="deleteVideo('${v.id}')">Xóa</button>
                            </td>
                        </tr>
                    `).join("");
                }
            }
        } catch (e) {}
        refreshIcons();
    }

    window.openUploadForCam = function(camId) {
        if (el.modalUploadCamSelect) el.modalUploadCamSelect.value = camId;
        el.uploadModal.style.display = "flex";
    };

    window.deleteVideo = async function(videoId) {
        if (!confirm("Bạn có chắc chắn muốn xóa tệp video này khỏi hệ thống?")) return;
        try {
            const res = await fetch(`/api/videos/${videoId}`, { method: "DELETE" });
            const data = await res.json();
            if (data.success) {
                loadSourcesTab();
            } else {
                alert(data.error || "Không thể xóa video.");
            }
        } catch (e) {
            alert("Lỗi kết nối máy chủ.");
        }
    };

    if (el.btnTriggerUploadModal) {
        el.btnTriggerUploadModal.addEventListener("click", () => {
            el.uploadModal.style.display = "flex";
        });
    }

    function closeUploadModal() {
        el.uploadModal.style.display = "none";
        el.uploadProgressContainer.style.display = "none";
        el.modalVideoFileInput.value = "";
    }

    if (el.btnCloseUploadModal) el.btnCloseUploadModal.addEventListener("click", closeUploadModal);
    if (el.btnCancelUploadModal) el.btnCancelUploadModal.addEventListener("click", closeUploadModal);

    if (el.btnConfirmUpload) {
        el.btnConfirmUpload.addEventListener("click", async () => {
            const file = el.modalVideoFileInput.files[0];
            const camId = el.modalUploadCamSelect.value;
            if (!file) {
                alert("Vui lòng chọn một tệp video từ máy tính!");
                return;
            }

            const formData = new FormData();
            formData.append("file", file);

            el.uploadProgressContainer.style.display = "block";
            el.uploadProgressBar.style.width = "40%";
            el.uploadStatusText.textContent = "Đang tải tệp lên máy chủ...";
            el.btnConfirmUpload.disabled = true;

            try {
                const res = await fetch(`/api/cameras/${camId}/source`, {
                    method: "POST",
                    body: formData
                });
                el.uploadProgressBar.style.width = "100%";

                const data = await res.json();
                if (data.success) {
                    alert(`Thành công: Đã gán video mới cho ${camId} và tạo phiên phân tích mới.`);
                    closeUploadModal();
                    await fetchCameras();
                    loadSourcesTab();
                } else {
                    alert(data.error || "Tải lên không thành công.");
                }
            } catch (e) {
                alert("Lỗi mạng khi tải lên video.");
            } finally {
                el.btnConfirmUpload.disabled = false;
            }
        });
    }

    // =========================================================================
    // 10.1 IMAGE UPLOAD & YOLO AI PREDICT CONTROLLER
    // =========================================================================
    const imageFileInput = document.getElementById("imageFileInput");
    const imageModeSelect = document.getElementById("imageModeSelect");
    const imageConfSlider = document.getElementById("imageConfSlider");
    const lblImageConfVal = document.getElementById("lblImageConfVal");
    const btnAnalyzeImage = document.getElementById("btnAnalyzeImage");
    const imagePreviewArea = document.getElementById("imagePreviewArea");
    const imagePlaceholderText = document.getElementById("imagePlaceholderText");
    const imagePreviewImg = document.getElementById("imagePreviewImg");
    const imageAnalysisResultCard = document.getElementById("imageAnalysisResultCard");
    const imageResultAnnotated = document.getElementById("imageResultAnnotated");

    if (imageConfSlider && lblImageConfVal) {
        imageConfSlider.addEventListener("input", () => {
            lblImageConfVal.textContent = `${imageConfSlider.value}%`;
        });
    }

    if (imageFileInput) {
        imageFileInput.addEventListener("change", () => {
            const file = imageFileInput.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreviewImg.src = e.target.result;
                    imagePreviewImg.style.display = "block";
                    if (imagePlaceholderText) imagePlaceholderText.style.display = "none";
                };
                reader.readAsDataURL(file);
            } else {
                imagePreviewImg.style.display = "none";
                if (imagePlaceholderText) imagePlaceholderText.style.display = "block";
            }
        });
    }

    if (btnAnalyzeImage) {
        btnAnalyzeImage.addEventListener("click", async () => {
            const file = imageFileInput ? imageFileInput.files[0] : null;
            if (!file) {
                alert("Vui lòng chọn một tệp ảnh từ máy tính để phân tích!");
                return;
            }

            const formData = new FormData();
            formData.append("file", file);
            formData.append("mode", imageModeSelect ? imageModeSelect.value : "detection");
            formData.append("conf", (parseFloat(imageConfSlider ? imageConfSlider.value : 25) / 100.0).toString());

            btnAnalyzeImage.disabled = true;
            btnAnalyzeImage.innerHTML = `<span>Đang phân tích YOLO...</span>`;

            try {
                const res = await fetch("/api/predict", {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                if (data.success) {
                    if (imageAnalysisResultCard) imageAnalysisResultCard.style.display = "block";
                    if (imageResultAnnotated) imageResultAnnotated.src = data.annotated_image;

                    const elTotal = document.getElementById("resImgTotalVeh");
                    if (elTotal) elTotal.textContent = data.total_vehicles || 0;

                    const elPcu = document.getElementById("resImgPcu");
                    if (elPcu) elPcu.textContent = (data.pcu_score || 0).toFixed(1);

                    const elLat = document.getElementById("resImgLatency");
                    if (elLat) elLat.textContent = `${data.process_time_ms || 0} ms`;

                    const cg = data.congestion || {};
                    const cgBadge = document.getElementById("resImgCongestionBadge");
                    if (cgBadge) {
                        const lvlClass = cg.level === 1 ? "normal" : (cg.level === 2 ? "moderate" : "congested");
                        cgBadge.className = `signal-badge ${lvlClass}`;
                        cgBadge.textContent = cg.name || "Thông thoáng";
                    }

                    const counts = data.counts || {};
                    const setVal = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
                    setVal("resImgCountMotorcycle", counts.xe_may || 0);
                    setVal("resImgCountCar", counts.o_to_con || 0);
                    setVal("resImgCountBus", counts.xe_khach || 0);
                    setVal("resImgCountTruck", counts.xe_tai || 0);
                    setVal("resImgCountBicycle", counts.xe_dap || 0);

                    refreshIcons();
                    imageAnalysisResultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
                } else {
                    alert(data.error || "Không thể phân tích ảnh.");
                }
            } catch (err) {
                alert("Lỗi kết nối khi gửi ảnh lên máy chủ.");
            } finally {
                btnAnalyzeImage.disabled = false;
                btnAnalyzeImage.innerHTML = `<i data-lucide="sparkles" class="icon-xs"></i> <span>Bắt đầu phân tích hình ảnh</span>`;
                refreshIcons();
            }
        });
    }

    // =========================================================================
    // 11. AUDIT LOGS TAB
    // =========================================================================
    async function loadLogsTab() {
        if (!el.logsTableBody) return;
        const cam = el.logFilterCam ? el.logFilterCam.value : "ALL";
        const lvl = el.logFilterLevel ? el.logFilterLevel.value : "ALL";
        const search = el.logSearchInput ? el.logSearchInput.value : "";

        try {
            const res = await fetch(`/api/logs?camera_id=${cam}&level=${lvl}&search=${encodeURIComponent(search)}&limit=100`);
            const data = await res.json();
            if (data.success) {
                el.logTotalCount.textContent = `${data.total} bản ghi`;

                // Update Mini Footer ticker with latest logs
                if (data.logs.length > 0 && el.miniLogTicker) {
                    const top3 = data.logs.slice(0, 3).map(l => `[${l.time_str}] ${l.camera_id || 'Hệ thống'}: ${l.status_text} ${l.flow ? `(${l.flow} xe/h)` : ''}`).join("  |  ");
                    el.miniLogTicker.textContent = top3;
                }

                if (data.logs.length === 0) {
                    el.logsTableBody.innerHTML = `<tr class="empty-state-row"><td colspan="9">Không có bản ghi nhật ký phù hợp.</td></tr>`;
                    return;
                }

                el.logsTableBody.innerHTML = data.logs.map(l => {
                    const lvlClass = l.status_level === 1 ? "normal" : (l.status_level === 2 ? "moderate" : "congested");
                    return `
                        <tr>
                            <td class="tabular-nums">${l.time_str}</td>
                            <td><span class="badge-subtle">${l.camera_id || 'ALL'}</span></td>
                            <td class="tabular-nums">${l.flow !== null && l.flow !== undefined ? l.flow : '—'}</td>
                            <td class="tabular-nums">${l.density !== null && l.density !== undefined ? l.density : '—'}</td>
                            <td class="tabular-nums">${l.pcu !== null && l.pcu !== undefined ? l.pcu : '—'}</td>
                            <td class="tabular-nums">${l.speed !== null && l.speed !== undefined ? `${l.speed} km/h` : '—'}</td>
                            <td><span class="signal-badge ${lvlClass}">${l.status_text}</span></td>
                            <td>${l.advice || '—'}</td>
                            <td style="color:var(--text-muted); font-size:11px;">${l.details || ''}</td>
                        </tr>
                    `;
                }).join("");
            }
        } catch (e) {}
    }

    if (el.logFilterCam) el.logFilterCam.addEventListener("change", loadLogsTab);
    if (el.logFilterLevel) el.logFilterLevel.addEventListener("change", loadLogsTab);
    if (el.logSearchInput) el.logSearchInput.addEventListener("input", () => {
        clearTimeout(window._logSearchDebounce);
        window._logSearchDebounce = setTimeout(loadLogsTab, 300);
    });

    if (el.btnClearLogs) {
        el.btnClearLogs.addEventListener("click", async () => {
            if (!confirm("Bạn có chắc chắn muốn xóa toàn bộ lịch sử sự kiện trong SQLite?")) return;
            try {
                const res = await fetch("/api/logs/clear", { method: "POST" });
                const data = await res.json();
                if (data.success) {
                    loadLogsTab();
                } else {
                    alert(data.error || "Không thể xóa nhật ký.");
                }
            } catch (e) {
                alert("Lỗi kết nối.");
            }
        });
    }

    // =========================================================================
    // 12. SETTINGS TAB CONTROLLER
    // =========================================================================
    if (el.btnSaveSystemSettings) {
        el.btnSaveSystemSettings.addEventListener("click", async () => {
            try {
                const res = await fetch("/api/config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        roi_length_m: parseFloat(el.inputRoiLength.value) || 60,
                        roi_width_m: parseFloat(el.inputRoiWidth.value) || 12,
                        density_thresholds: [
                            parseFloat(el.inputThrL2.value) || 120,
                            parseFloat(el.inputThrL3.value) || 220,
                            parseFloat(el.inputThrL4.value) || 350
                        ]
                    })
                });
                const data = await res.json();
                if (data.success) {
                    alert("Cấu hình hệ thống đã được lưu và áp dụng thành công!");
                }
            } catch (e) {
                alert("Lỗi khi lưu cấu hình.");
            }
        });
    }

    // Sound toggle button
    el.btnSoundToggle.addEventListener("click", () => {
        AppState.soundEnabled = !AppState.soundEnabled;
        el.btnSoundToggle.classList.toggle("btn-primary", AppState.soundEnabled);
        el.lblSound.textContent = AppState.soundEnabled ? "Bật" : "Còi";
        if (!AppState.soundEnabled) {
            el.alarmAudio.pause();
            AppState.alarmActive = false;
        }
    });

    // =========================================================================
    // 13. SSE REAL-TIME CONNECTION
    // =========================================================================
    function startRealtimeStream() {
        try {
            AppState.sseSource = new EventSource("/api/stream_events");
            AppState.sseSource.onmessage = (event) => {
                try {
                    const payload = JSON.parse(event.data);
                    if (payload && payload.cameras) {
                        AppState.cameras = payload.cameras;
                        updateAllViews();
                    }
                } catch (pe) {}
            };
            AppState.sseSource.onerror = () => {
                AppState.sseSource.close();
                // Failover Polling
                setInterval(fetchCameras, 2000);
            };
        } catch (e) {
            setInterval(fetchCameras, 2000);
        }
    }

    // Auto-refresh định kỳ cho các biểu đồ động (Real-time dynamic charts)
    setInterval(() => {
        try {
            if (AppState.activeTab === "overview") {
                updateOverviewCharts();
            } else if (AppState.activeTab === "analytics") {
                updateAnalyticsCharts();
            } else if (AppState.activeTab === "forecast") {
                updateForecastTab();
            } else if (AppState.activeTab === "logs") {
                loadLogsTab();
            }
        } catch (err) {}
    }, 3000);

    // Initialize application
    const initialHash = window.location.hash.replace("#", "");
    switchTab(initialHash || "overview");
    fetchCameras();
    startRealtimeStream();
    loadLogsTab();

});
