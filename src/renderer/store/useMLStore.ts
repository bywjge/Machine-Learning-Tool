import { defineStore } from "pinia";
import { Ref, ref } from "vue";
import { ModelCluster} from "../../model/ModelCluster";
//功能
let prepareData = () => {
    let result = [];
    let model = new ModelCluster();
    model.mainName = "数据导入";
    model.rightCorner = "";
    model.subName = "在这里导入、查看数据";
    model.avatar = `http://127.0.0.1:5000/cluster/get-icon?img=load_data&t=1`;
    result.push(model);
    model = new ModelCluster();
    model.mainName = "模型选择";
    model.rightCorner = "";
    model.subName = "Resnet、VGG...";
    model.avatar = `http://127.0.0.1:5000/cluster/get-icon?img=model&t=1`;
    result.push(model);
    model = new ModelCluster();
    model.mainName = "模型训练";
    model.rightCorner = "";
    model.subName = "设置迭代次数、学习率";
    model.avatar = `http://127.0.0.1:5000/cluster/get-icon?img=model-training&t=1`;
    result.push(model);
    return result;
};
//定义一个Store
export const useMLStore = defineStore("ML", () => {
    let data: Ref<ModelCluster[]> = ref(prepareData());
    let selectItem = (item: ModelCluster) => {
        if (item.isSelected) return;
        data.value.forEach((v) => (v.isSelected = false));
        item.isSelected = true;

    };
    return { data, selectItem };
});
