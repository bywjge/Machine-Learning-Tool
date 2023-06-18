import { app, BrowserWindow } from "electron";
import {CustomScheme} from "./CustomScheme";
import {CommonWindowEvent} from "./CommonWindowEvent";
process.env.ELECTRON_DISABLE_SECURITY_WARNINGS = "true";
let mainWindow: BrowserWindow;
app.on("browser-window-created", (e, win) => {
    CommonWindowEvent.regWinEvent(win);
});
app.whenReady().then(() => {

    let config = {
        // transparent:true, // 设置窗口为透明
        // frame选项用于控制是否显示应用程序的窗口装饰（例如标题栏、边框等）。当frame设置为false时，将不会显示窗口装饰。
        frame: false,
        // 默认初始大小
        width: 1000,
        webPreferences: {
            nodeIntegration: true,
            webSecurity: false,
            allowRunningInsecureContent: true,
            contextIsolation: false,
            webviewTag: true,
            spellcheck: false,
            disableHtmlFullscreenWindowResize: true,
        },
    };
    mainWindow = new BrowserWindow(config);
    // mainWindow.setOpacity(0.95); // 设置透明度为50%
    mainWindow.webContents.openDevTools({ mode: "undocked" });
    if (process.argv[2]) {
        mainWindow.loadURL(process.argv[2]);
    } else {
        CustomScheme.registerScheme();
        mainWindow.loadURL(`app://index.html`);
    }

    CommonWindowEvent.listen();
    CommonWindowEvent.regWinEvent(mainWindow);
});
