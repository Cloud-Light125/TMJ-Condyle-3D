# TMJ Condyle Annotator

这是一个面向牙医和医学生的中文 3D Slicer 标注工作台。模块把 Segment Editor
作为内部编辑引擎嵌入自己的页面，用户不需要切换到 Slicer 原生 Segment Editor，
也不需要理解 MRML、Python 或其它 Slicer 内部概念。

本项目的标注定义固定为：

    0 = background
    1 = mandibular_condyle

## 使用流程

1. 在 3D Slicer 中通过 Edit > Application Settings > Modules > Additional module
   paths 添加本目录，然后重启或刷新模块。
2. 搜索并打开“下颌髁突三维标注”。英文模块名 `TMJ Condyle Annotator` 只用于
   Slicer 内部识别和“关于”信息。
3. 在“导入核磁”页面选择 MRI，或使用已经加载的当前病例。
4. 在“标注下颌髁突”页面使用中文“画笔”和“擦除”，按主要标注视图逐层检查。
5. 点击“标完了，下一步：检查”，在三视图和 3D 视图中检查轮廓。
6. 技术检查通过后进入“保存本例”，保存本例的二值 Mask。

工作台只允许一个 segment，用户可见名称为“下颌髁突”，内部名称固定为
`Mandibular Condyle`。常用工具只有画笔和擦除；剪刀、清理零碎区域、层间补全
位于“辅助工具”折叠区域。

保存前模块会检查 geometry、0/1 标签和非空前景，并在技术检查失败时阻止保存。
导出 mask 不会被模块自动修正。导出后如需更新项目 manifest，请确认私有的
`workspace/dataset_manifest.csv` 路径和病例状态。

## 退出简洁模式

模块默认临时隐藏与标注无关的 Slicer 工具栏和开发者停靠面板，并保留必要窗口
控制。点击右上角“退出简洁模式”可恢复完整 Slicer；离开模块后会自动恢复原状，
不会永久修改用户的 Slicer 配置。
