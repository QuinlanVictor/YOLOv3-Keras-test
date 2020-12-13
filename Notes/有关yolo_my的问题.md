#### 20201212

解决方案代码：

        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match

因为保存的时候是model.save_weights，所以需要提前构建模型

不知道为什么11号的训练出了问题，所以修改了一下代码，改成以上的格式，能够正常运行

记录一下这个问题
