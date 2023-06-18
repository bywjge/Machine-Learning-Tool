<script setup lang="ts">
import { ipcRenderer } from "electron";
import { onMounted } from "vue";
import BarLeft from "../Component/BarLeft.vue";
onMounted(() => {
  ipcRenderer.invoke("showWindow");
});
</script>

<template>
  <BarLeft />
  <div class="pageBox">
    <router-view />
  </div>
</template>

<style scoped lang="scss">
.pageBox {
  flex: 1;
  height: 100%;
  border-top: 1px solid #e6e6e6;
  box-sizing: border-box;
  display: flex;
  // 2023.6.10 添加，导入数据框要加这个才不会有滚动条
  overflow-x: auto;
  margin-top: -1px;
}
</style>
