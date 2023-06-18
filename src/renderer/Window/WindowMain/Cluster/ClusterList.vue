<script setup lang="ts">
import {onMounted} from "vue";
import ListItem from "./ListItem.vue";
import {useClusterStore} from "../../../store/useClusterStore";

let store = useClusterStore();

onMounted(() => {
  //选中第1个功能
  store.selectItem(store.data[0]);
});
</script>
<template>
  <div class="ChatList">
<!--    <ChatSearch/>-->
    <div class="ListBox">
      <ListItem :data="item" v-for="item in store.data" :key="item.id"/>
    </div>
  </div>
</template>
<style scoped lang="scss">
.ChatList {
  font-family: apple,serif;
  //width: 250px;
  width: 220px;
  display: flex;
  flex-direction: column;
  height: 100%;
  //overflow-y: auto;
  box-sizing: border-box;
}

.ListBox {
  background: rgb(230, 229, 229) linear-gradient(to bottom right, rgb(235, 234, 233), rgb(240, 240, 240));
  flex: 1;
  overflow-y: auto;
  box-sizing: border-box;

  border-right: 1px solid rgb(214, 214, 214);
}
</style>
