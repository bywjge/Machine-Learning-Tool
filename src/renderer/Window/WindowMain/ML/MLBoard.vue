<script setup lang="ts">
import {QuestionCircleOutlined} from '@ant-design/icons-vue';
import {computed, ref} from "vue";
import {useMLStore} from "../../../store/useMLStore";
import BarTop from "../../../Component/BarTop.vue";
import {ipcRenderer} from "electron";

let store = useMLStore();
let tipVisible = ref(false);

let dataDirPath = ref('');
let tableData = ref([]);

let trainPercent = ref(70);
let testPercent = ref(30);
let batchSize = ref(32);
let chooseModel = ref('');
let chooseOptimizer = ref('Adam');
let epoch = ref(20);
let learningRate = ref(0.0001)

let dataDivisionStatus = ref('未划分');
let modelSelectStatus = ref('未选择模型');

let showMLMessage = ref([['blue', '等待训练开始...']]);

let barTitle = computed(() => {
  const [selectedData] = store.data.filter(item => item.isSelected);
  return selectedData ? selectedData.mainName : "";
})

const tableColumn = [
  {
    title: '文件夹名(数据标签)',
    dataIndex: 'label',
    key: 'label',
  },
  {
    title: '文件数量(数据量)',
    dataIndex: 'amount',
    key: 'amount',
  },
]


const getPath = () => {
  ipcRenderer.send('open-directory-dialog', 'openDirectory');
  ipcRenderer.on('selectedItem', function (e, files) {
    console.log("getPath函数" + files)
    dataDirPath.value = files;
    // submitDataDirPath();
  });

}

const submitDataDirPath = () => {
  console.log("submitDataDirPath提交数据")
  // 轮询
  fetch('http://127.0.0.1:5000/ml/upload-path', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      dirPath: dataDirPath.value,
    }),
  })
      .then(response => response.json())
      .then(data => {
        console.log(data); // Handle the retrieved JSON data here
        tableData.value = data.data;
      })
      .catch(error => {
        console.error('Error:', error);
      });
}

const submitDataDivision = () => {
  console.log("submitDataDivision提交数据")
  // 轮询
  fetch('http://127.0.0.1:5000/ml/load-dataset', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      train_test_ratio: trainPercent.value / 100,
      batch_size: batchSize.value,
    }),
  })
      .then(response => response.json())
      .then(data => {
        console.log(data); // Handle the retrieved JSON data here
        dataDivisionStatus.value = '划分成功';
      })
      .catch(error => {
        console.error('Error:', error);
        dataDivisionStatus.value = '划分失败';
      });
}

const confirmModel = () => {
  console.log("confirmModel提交数据")
  modelSelectStatus.value = '正在下载模型';
  // 轮询
  fetch('http://127.0.0.1:5000/ml/model-select', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: chooseModel.value,
    }),
  })
      .then(response => response.json())
      .then(data => {
        console.log(data); // Handle the retrieved JSON data here
        modelSelectStatus.value = '模型选择成功';
      })
      .catch(error => {
        console.error('Error:', error);
        modelSelectStatus.value = '模型选择失败';
      });
}

const startTraining = () => {
  // 轮询
  let temp = setInterval(refreshMLMessage, 500);
  fetch('http://127.0.0.1:5000/ml/model-train', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({epoch: epoch.value, optimizer: chooseOptimizer.value, learningRate: learningRate.value}),
  })
      .then(response => response.json())
      .then(data => {
        console.log(data); // Handle the retrieved JSON data here

        clearInterval(temp);
      })
      .catch(error => {
        console.error('Error:', error);
      });
}

const refreshMLMessage = () => {
  fetch('http://127.0.0.1:5000/ml/get-info')
      .then(response => response.json())
      .then(data => {
        // 在这里处理获取到的数据
        console.log(data);
        showMLMessage.value = data.info;
        // 可以将获取到的数据进行进一步的处理或展示
      })
      .catch(error => {
        // 处理错误情况
        console.error('Error:', error);
      });
}

</script>

<template>
  <div class="Board">
    <BarTop :title="barTitle"/>
    <div v-if="store.data[0].isSelected" class="MessageList">
      <a-modal v-model:visible="tipVisible" title="Help" @ok="()=>{tipVisible = false}">
        <p>Some contents...</p>
        <p>Some contents...</p>
        <p>Some contents...</p>
      </a-modal>
      <!--      <a-button class="MyButton" @click="getPath">选择文件夹</a-button>-->


      <div style="display: flex; align-items: center;padding-right: 20px">
        <a-input-search style="padding: 20px; flex: 1;"
                        v-model:value="dataDirPath"
                        placeholder="点击右侧选择文件夹"
                        enter-button="选择文件夹"
                        size="large"
                        @search="getPath"

        />
        <a-button @click="()=>{tipVisible = true}" shape="circle">
          <template #icon>
            <QuestionCircleOutlined/>
          </template>
        </a-button>
      </div>

      <a-button @click="submitDataDirPath" class="MyButton">上传</a-button>

      <a-table :columns="tableColumn" :data-source="tableData" bordered>
        <template #bodyCell="{ column, text }">
          <!--          <template v-if="column.dataIndex === 'name'">-->
          <!--            <a>{{ text }}</a>-->
          <!--          </template>-->
        </template>
        <template #title>当前路径下的子文件夹</template>
        <!--        <template #footer>Footer</template>-->
      </a-table>


      <!--      <a-button @click="submitDataDirPath" class="MyButton">确定</a-button>-->


<!--      {{ dataDirPath }}-->
    </div>

    <div v-if="store.data[1].isSelected" class="MessageList">
      <!--      模型训练 选择模型时有介绍-->
      <div style="padding-top: 10px;padding-left: 10px">
        <h2>数据划分</h2>
        训练集占比
        <a-input-number id="inputTrainNumber" v-model:value="trainPercent" :min="1" :max="99"/>
        %
        <br>
        测试集占比
        <a-input-number id="inputTestNumber" :value="100 - trainPercent" :min="1" :max="99"/>
        %
        <br>
        批次大小
        <a-input-number id="batchSize" v-model:value="batchSize" :min="1" :max="99"/>

        <a-button type="primary" class="MyButton" @click="submitDataDivision">划分数据集</a-button>

        {{ dataDivisionStatus }}
      </div>
      <a-divider/>
      <div style="padding-top: 10px;padding-left: 10px">
        <h2>模型选择</h2>
        <a-select
            ref="select"
            v-model:value="chooseModel"
            style="width: 600px"
        >
          <a-select-option value="ResNet-18">ResNet-18</a-select-option>
          <a-select-option value="ResNet-50">ResNet-50</a-select-option>
          <a-select-option value="VGG" disabled>VGG</a-select-option>
        </a-select>
        <div v-if="chooseModel==='ResNet-18'" style="padding-top: 10px">
          <a-card title="ResNet-18" style="width: 600px">
            <!--          <template #extra><a href="#">more</a></template>-->
            <p>ResNet-18是由微软研究院提出的一个深度残差网络模型。</p>
            <p>它具有18个卷积层，包括卷积层、批归一化层和全连接层。</p>
            <p>ResNet-18通过使用残差块结构来解决梯度消失问题，使得更深层次的网络能够更容易地训练和优化。</p>
            <p>它在图像分类等计算机视觉任务中表现出色，是一个广泛使用的预训练模型。</p>
          </a-card>
        </div>
        <div v-if="chooseModel==='ResNet-50'" style="padding-top: 10px">
          <a-card title="ResNet-50" style="width: 600px">
            <!--          <template #extra><a href="#">more</a></template>-->
            <p>ResNet-50是ResNet系列中更深更复杂的一个模型。</p>
            <p>它具有50个卷积层，包括卷积层、批归一化层和全连接层。</p>
            <p>ResNet-50相比于ResNet-18更深，可以学习到更多抽象的特征表示。</p>
            <p>它在许多计算机视觉任务中取得了显著的成果，包括图像分类、目标检测和语义分割等。</p>
          </a-card>
        </div>
        <a-button type="primary" class="MyButton" @click="confirmModel">选择该模型</a-button>


<!--        TODO：是否预训练-->
        {{modelSelectStatus}}

      </div>

<!--      {{chooseModel}}-->


<!--      <a-divider/>-->



    </div>

    <div v-if="store.data[2].isSelected" class="MessageList">
      <div style="padding-top: 10px;padding-left: 10px;padding-bottom: 30px">
<!--        <h2>模型训练</h2>-->
        迭代次数
        <a-input-number id="epoch" v-model:value="epoch" :min="1" :max="200"/>
        <div>优化器选择</div>
        <a-select
            ref="select"
            v-model:value="chooseOptimizer"
            style="width: 600px"
        >
          <a-select-option value="Adam">Adam</a-select-option>
          <a-select-option value="SGD" disabled>SGD</a-select-option>
          <a-select-option value="more" disabled>(目前仅支持Adam)待完善......</a-select-option>
        </a-select>
        <div>学习率
          <a-input-number id="learningRate" v-model:value="learningRate" :min="0.000001" :max="0.1"/></div>
        <a-button type="primary" class="MyButton" @click="startTraining">开始训练</a-button>
        <a-timeline>

          <a-timeline-item v-for="item in showMLMessage" :key="item" :color="item[0] ">{{ item[1] }}</a-timeline-item>
        </a-timeline>
      </div>
<!--      {{epoch}}-->
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
