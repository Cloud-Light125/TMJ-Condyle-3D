# TMJ Condyle Annotator

这是一个面向牙医和医学生的中文 3D Slicer 标注工作台。模块把 Segment Editor
作为内部编辑引擎嵌入自己的页面，用户不需要切换到 Slicer 原生 Segment Editor，
也不需要理解 MRML、Python 或其它 Slicer 内部概念。

本项目的标注定义固定为：

    0 = background
    1 = mandibular_condyle

## 使用流程

1. 双击项目根目录的“启动下颌髁突标注.bat”。它会自动打开 Slicer、加载模块，
   并直接进入本工作台。
2. 首页点击“开始新的标注”，选择核磁文件或导入病例文件夹。
3. 用中文“画笔”和“擦除”逐层检查，再点击“标完了，检查三维”。
4. 在 3D 检查页确认形状后保存，并继续下一例。

工作台只允许一个 segment，用户可见名称为“下颌髁突”，内部 ID 固定为
`MandibularCondyle`。常用工具只有画笔和擦除；剪刀、清理零碎区域、层间补全
位于“辅助工具”折叠区域。

保存前模块会检查 geometry、0/1 标签和非空前景，并在技术检查失败时阻止保存。
导出 mask 不会被模块自动修正。导出后如需更新项目 manifest，请确认私有的
`workspace/dataset_manifest.csv` 路径和病例状态。

## 开发者 / 高级使用

如果不能使用一键启动器，才需要在 Slicer 的 Additional module paths 中手工添加本目录。
启动脚本 `slicer/startup_tmj.py` 负责等待模块注册后自动进入工作台。

## 退出简洁模式

模块默认临时隐藏与标注无关的 Slicer 工具栏和开发者停靠面板，并保留必要窗口
控制。点击右上角“显示完整 Slicer”可恢复完整 Slicer；离开模块后会自动恢复原状，
不会永久修改用户的 Slicer 配置。
