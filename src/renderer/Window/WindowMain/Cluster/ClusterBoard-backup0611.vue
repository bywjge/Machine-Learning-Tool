<script setup lang="ts">
import BarTop from "../../../Component/BarTop.vue";
import {UploadOutlined} from '@ant-design/icons-vue';
import {message, UploadChangeParam} from "ant-design-vue";
import {ref} from "vue";
import {computed} from "vue";
import {useClusterStore} from "../../../store/useClusterStore";

let store = useClusterStore();
let returnData = ref([]);
let fileList = ref([]);
let selectTags = ref([]);
let columns = ref([
  {
    title: '未上传数据',
    dataIndex: 'number',
    key: 'number',
  },
  // {
  //   title: '姓名',
  //   dataIndex: 'name',
  //   key: 'name',
  //   // width: 300,
  // },
]);
let tagOptions = ref([{value: "请先导入数据", disabled: true}]);
let showMessage = ref([['blue', '等待提交...']]);
let showImage = ref<boolean>(false);
let imgSrc = ref<string>('');

let barTitle = computed(() => {
  const [selectedData] = store.data.filter(item => item.isSelected);
  return selectedData ? selectedData.mainName : "";
})

// const selectedObject = store.data.find(item => item.isSelected === true);
// const selectedMainName = selectedObject ? selectedObject.mainName : '';
// barTitle.value = selectedMainName;
// 上传数据集
const handleUploadChange = (info: UploadChangeParam) => {
  if (info.file.status !== 'uploading') {
    console.log(info.file, info.fileList);
  }
  if (info.file.status === 'done') {
    message.success(`${info.file.name} 上传成功`);
    fetch('http://127.0.0.1:5000/cluster/data-header')
        .then(response => response.json())
        .then(data => {
          // data = [data];
          // tagOptions.value = data;
          console.log(tagOptions);

          // 选择哪些属性进行聚类
          tagOptions.value = data.map((item) => {
            return {
              value: item,
            };
          });

          // 表格列名
          columns.value = data.map((item) => {
            return {
              title: item,
              dataIndex: item,
              key: item,
            };
          });
          console.log(columns); // 处理返回的csv标题头
        })
        .catch(error => {
          console.error('Error:', error);
        });
    fetch('http://127.0.0.1:5000/cluster/data')
        .then(response => response.json())
        .then(data => {
          returnData.value = data;
          console.log(data); // Handle the retrieved JSON data here
        })
        .catch(error => {
          console.error('Error:', error);
        });

  } else if (info.file.status === 'error') {
    message.error(`${info.file.name} 上传失败`);
  }
};

const handleChange = (value: string) => {
  console.log(`selected ${value}`);
};

// 清空表格
const clearTable = () => {
  returnData.value = [];
  fileList.value = [];
  selectTags.value = [];
  tagOptions.value = [{value: "请先导入数据", disabled: true}];
  showMessage.value = [['blue', '等待提交...']];
  showImage.value = false;
  columns.value = [
    {
      title: '未上传数据',
      dataIndex: 'number',
      key: 'number',
    }]
  console.log('重置成功');
};

// 刷新文字提示
const refreshMessage = () => {
  fetch('http://127.0.0.1:5000/cluster/get-info')
      .then(response => response.json())
      .then(data => {
        // 在这里处理获取到的数据
        console.log(data);
        // showMessage.value = data.info.replace(/\n/g, '<br>');
        showMessage.value = data.info;
        // 可以将获取到的数据进行进一步的处理或展示
      })
      .catch(error => {
        // 处理错误情况
        console.error('Error:', error);
      });
}

// 提交聚类的属性
const submitTags = () => {
  console.log("提交的数据:" + JSON.stringify({tags: selectTags.value}))
  // 轮询
  let temp = setInterval(refreshMessage, 500);
  fetch('http://127.0.0.1:5000/cluster/upload-tag', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({tags: selectTags.value}),
  })
      .then(response => response.json())
      .then(data => {
        console.log(data); // Handle the retrieved JSON data here
        // 显示图片
        // 避免缓存
        const randomParam = Date.now(); // 或者使用其他的随机字符串生成方法
        imgSrc.value = `http://127.0.0.1:5000/cluster/get-img?random=${randomParam}`;
        showImage.value = true;
        clearInterval(temp);
      })
      .catch(error => {
        console.error('Error:', error);
      });
};

</script>

<template>
  <div class="Board">
    <!--    <BarTop title="聚类算法"/>-->
    <BarTop :title=barTitle />
    <div v-if="store.data[0].isSelected" class="MessageList">

      <a-button class="MyButton" type="primary" @click="clearTable">重置</a-button>
      <a-upload
          v-model:file-list="fileList"
          name="file"
          action="http://127.0.0.1:5000/cluster/upload-csv"
          @change="handleUploadChange"
      >

        <a-button class="MyButton">
          <upload-outlined></upload-outlined>
          上传数据集
        </a-button>

      </a-upload>

      <a-divider/>
      <!--      <div>-->
      <a-table :columns="columns" :data-source="returnData"
               :scroll="{ x: 'max-content' }">
        <!--          <template #bodyCell="{ column, text }">-->
        <!--            <template v-if="column.dataIndex === 'name'">-->
        <!--              <a>{{ text }}</a>-->
        <!--            </template>-->
        <!--          </template>-->
      </a-table>

      <!--      </div>-->

      {{ store.data }}
      <!--      测试显示用  -->
      <!--      {{columns}}-->
      <!--      {{ returnData }}-->
      <!--      1234566625423-->

    </div>

    <div v-if="store.data[1].isSelected" class="MessageList">

      <div class="K-meanBox">
        ① 请选择需要聚类的属性：

        <a-select class="TagSelect"
                  v-model:value="selectTags"
                  mode="tags"
                  style="width: 100%"
                  placeholder="Tags Mode"
                  :options="tagOptions"
                  @change="handleChange"
        ></a-select>
        <a-button class="MyButton" type="primary" @click="submitTags">提交</a-button>

        <a-timeline>
          <!--          <a-timeline-item>Create a services site 2015-09-01</a-timeline-item>-->
          <!--          <a-timeline-item>Solve initial network problems 2015-09-01</a-timeline-item>-->
          <!--          <a-timeline-item>Technical testing 2015-09-01</a-timeline-item>-->
          <!--          <a-timeline-item color="green">Network problems being solved 2015-09-01</a-timeline-item>-->
          <!--         item[0]是颜色 item[1]是信息-->
          <a-timeline-item v-for="item in showMessage" :key="item" :color="item[0] ">{{ item[1] }}</a-timeline-item>
        </a-timeline>

        <a-image
            v-if="showImage"
            :width="600"
            :src="imgSrc"
        />
      </div>
      {{ showMessage }}
      <!--      <div v-html="showMessage"></div>-->

    </div>
  </div>
</template>


<style scoped lang="scss">

.K-meanBox {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  background: rgb(245, 245, 245);
  padding-top: 15px;
  padding-left: 10px;
}

.TagSelect {
  padding-left: 20px;
  padding-right: 20px;
}


.MyButton {
  //position: center;
  margin: 20px;
  border-radius: 8px;
}


.Board {
  font-family: apple, serif;
  height: 100%;
  display: flex;
  flex: 1;
  // 这里也要加
  overflow-x: auto;
  flex-direction: column;
}

.MessageList {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  background: rgb(245, 245, 245);
}
</style>
