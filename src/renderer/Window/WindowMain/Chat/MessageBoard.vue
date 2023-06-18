<script setup lang="ts">
import BarTop from "../../../Component/BarTop.vue";
import {ref, defineComponent} from "vue";
import {useChatStore} from "../../../store/useChatStore";
import {useMessageStore} from "../../../store/useMessageStore";
import {ModelChat} from "../../../../model/ModelChat";
import MessageItem from "./MessageItem.vue";
import {message, UploadChangeParam} from "ant-design-vue";

let store = useChatStore();
let messageStore = useMessageStore();
let curId = "";
let returnData = ref([]);

const handleChange = (info: UploadChangeParam) => {
  if (info.file.status !== 'uploading') {
    console.log(info.file, info.fileList);
  }
  if (info.file.status === 'done') {
    message.success(`${info.file.name} file uploaded successfully`);
    fetch('http://127.0.0.1:5000/data')
        .then(response => response.json())
        .then(data => {
          returnData.value = data;
          console.log(data); // Handle the retrieved JSON data here
        })
        .catch(error => {
          console.error('Error:', error);
        });

  } else if (info.file.status === 'error') {
    message.error(`${info.file.name} file upload failed.`);
  }
};


const handleClick = (info: UploadChangeParam) => {
  returnData.value = [];
  console.log(111);
};
const fileList = ref([]);

const columns = [
  {
    title: '序号',
    dataIndex: 'number',
    key: 'number',
  },
  {
    title: '姓名',
    dataIndex: 'name',
    key: 'name',
    width: 300,
  },
];

</script>
<template>
  <div class="MessageBord">
    <BarTop/>
    <div class="MessageList">
      <a-button class="MyButton" type="primary" @click="handleClick">清空表格</a-button>
      <a-upload
          v-model:file-list="fileList"
          name="file"
          action="http://127.0.0.1:5000/cluster/upload-csv"
          :headers="headers"
          @change="handleChange"
      >
        <a-button>
          <upload-outlined></upload-outlined>
          点击上传
        </a-button>
      </a-upload>

      <a-table :columns="columns" :data-source="returnData">
        <template #bodyCell="{ column, text }">
          <template v-if="column.dataIndex === 'name'">
            <a>{{ text }}</a>
          </template>
        </template>
      </a-table>
      {{ returnData }}


      <!--      <MessageItem :data="item" v-for="item in messageStore.data" :key="item.id" />-->
      1234566625423
    </div>
  </div>
</template>
<style scoped lang="scss">
.MyButton {
  position: center;
  border-radius: 8px;
}

.MessageBord {
  font-family: apple;
  height: 100%;
  display: flex;
  flex: 1;
  flex-direction: column;
}

.MessageList {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  background: rgb(245, 245, 245);
}
</style>
