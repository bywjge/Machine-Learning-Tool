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
    model.mainName = "聚类";
    model.rightCorner = "";
    model.subName = "K-means聚类";
    model.avatar = `http://127.0.0.1:5000/cluster/get-icon?img=cluster&t=1`;
    result.push(model);
    model = new ModelCluster();
    model.mainName = "降维";
    model.rightCorner = "";
    model.subName = "PCA降维";
    model.avatar = `http://127.0.0.1:5000/cluster/get-icon?img=dimension&t=1`;
    result.push(model);
    return result;
};
//定义一个Store
export const useClusterStore = defineStore("cluster", () => {
    let data: Ref<ModelCluster[]> = ref(prepareData());
    let selectItem = (item: ModelCluster) => {
        if (item.isSelected) return;
        data.value.forEach((v) => (v.isSelected = false));
        item.isSelected = true;

    };
    return { data, selectItem };
});
