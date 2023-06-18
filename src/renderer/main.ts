import {createApp} from 'vue';
import "./assets/style.css";
import "./assets/icon/iconfont.css";
import "./assets/icon/iconfont1.css";
import './assets/font/index.css';
import Antd from 'ant-design-vue';
import App from './App.vue';
import {router} from "./router";
import {createPinia} from "pinia";
import 'ant-design-vue/dist/antd.css'
// import '@surely-vue/table/dist/index.less';
import STable from '@surely-vue/table';
// import '@surely-vue/table/src/style/antdv.less';

createApp(App)
    // .use(STable)
    .use(createPinia())
    .use(router)
    .use(Antd)
    .mount("#app");
