import { ModelBase } from "./ModelBase";
export class ModelCluster extends ModelBase{
    // 功能名称
    mainName?: string;
    // 副标题
    subName?: string;
    // 右上角
    rightCorner?: number | string;
    // 是否被选中
    isSelected = false;

    avatar?: string;
}
