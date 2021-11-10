import torch
from dataloader import DataSet
from torch.utils.data import DataLoader

from MaskRcnn import resnet50_fpn_backbone, FasterRCNN, FastRCNNPredictor


# 33
def create_model(num_classes, device):
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     trainable_layers=3)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = FasterRCNN(backbone=backbone, num_classes=91)
    # 载入预训练模型权重
    # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    weights_dict = torch.load(r"J:\Beijing\Sartorius\fasterrcnn_resnet50_fpn_coco.pth", map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

'''
detectron2
'''
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                                kernel_size=1, stride=1, bias=False)  # squeeze channels
#         self.bn1 = norm_layer(out_channel)
#         # -----------------------------------------
#         self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#                                kernel_size=3, stride=stride, bias=False, padding=1)
#         self.bn2 = norm_layer(out_channel)
#         # -----------------------------------------
#         self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
#                                kernel_size=1, stride=1, bias=False)  # unsqueeze channels
#         self.bn3 = norm_layer(out_channel * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, blocks_num, num_classes=1000, include_top=True, norm_layer=None):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.include_top = include_top
#         self.in_channel = 64
#
#         self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = norm_layer(self.in_channel)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#
#         # 数字是ResNet各layer的输入通道维度
#         self.layer1 = self._make_layer(block, 64, blocks_num[0])
#         self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
#         if self.include_top:
#             self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
#             self.fc = nn.Linear(512 * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#
#     def _make_layer(self, block, channel, block_num, stride=1):
#         norm_layer = self._norm_layer
#         downsample = None
#         if stride != 1 or self.in_channel != channel * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 norm_layer(channel * block.expansion))
#
#         layers = []
#         layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, norm_layer=norm_layer))
#         self.in_channel = channel * block.expansion
#
#         for _ in range(1, block_num):
#             layers.append(block(self.in_channel, channel, norm_layer=norm_layer))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         if self.include_top:
#             x = self.avgpool(x)
#             x = torch.flatten(x, 1)
#             x = self.fc(x)
#
#         return x
#
#
# class FasterRCNNBase(nn.Module):
#     """
#     Main class for Generalized R-CNN.
#
#     Arguments:
#         backbone (nn.Module):
#         rpn (nn.Module):
#         roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
#             detections / masks from it.
#         transform (nn.Module): performs the data transformation from the inputs to feed into
#             the model
#     """
#
#     def __init__(self, backbone, rpn, roi_heads, transform):
#         super(FasterRCNNBase, self).__init__()
#         self.transform = transform
#         self.backbone = backbone
#         self.rpn = rpn
#         self.roi_heads = roi_heads
#         # used only on torchscript mode
#         self._has_warned = False
#
#     @torch.jit.unused
#     def eager_outputs(self, losses, detections):
#         # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
#         if self.training:
#             return losses
#
#         return detections
#
#     def forward(self, images, targets=None):
#         # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
#         """
#         Arguments:
#             images (list[Tensor]): images to be processed
#             targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
#
#         Returns:
#             result (list[BoxList] or dict[Tensor]): the output from the model.
#                 During training, it returns a dict[Tensor] which contains the losses.
#                 During testing, it returns list[BoxList] contains additional fields
#                 like `scores`, `labels` and `mask` (for Mask R-CNN models).
#
#         """
#         if self.training and targets is None:
#             raise ValueError("In training mode, targets should be passed")
#
#         if self.training:
#             assert targets is not None
#             for target in targets:  # 进一步判断传入的target的boxes参数是否符合规定
#                 boxes = target["boxes"]
#                 if isinstance(boxes, torch.Tensor):
#                     if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
#                         raise ValueError("Expected target boxes to be a tensor"
#                                          "of shape [N, 4], got {:}.".format(
#                             boxes.shape))
#                 else:
#                     raise ValueError("Expected target boxes to be of type "
#                                      "Tensor, got {:}.".format(type(boxes)))
#
#         original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
#         for img in images:
#             val = img.shape[-2:]
#             assert len(val) == 2  # 防止输入的是个一维向量
#             original_image_sizes.append((val[0], val[1]))
#         # original_image_sizes = [img.shape[-2:] for img in images]
#
#         images, targets = self.transform(images, targets)  # 对图像进行预处理
#
#         # print(images.tensors.shape)
#         features = self.backbone(images.tensors)  # 将图像输入backbone得到特征图
#         if isinstance(features, torch.Tensor):  # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
#             features = OrderedDict([('0', features)])  # 若在多层特征层上预测，传入的就是一个有序字典
#
#         # 将特征层以及标注target信息传入rpn中
#         # proposals: List[Tensor], Tensor_shape: [num_proposals, 4],
#         # 每个proposals是绝对坐标，且为(x1, y1, x2, y2)格式
#         proposals, proposal_losses = self.rpn(images, features, targets)
#
#         # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分
#         detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
#
#         # 对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
#         detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
#
#         losses = {}
#         losses.update(detector_losses)
#         losses.update(proposal_losses)
#
#         if torch.jit.is_scripting():
#             if not self._has_warned:
#                 warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
#                 self._has_warned = True
#             return losses, detections
#         else:
#             return self.eager_outputs(losses, detections)
#
#         # if self.training:
#         #     return losses
#         #
#         # return detections
#
#
# class FasterRCNN(FasterRCNNBase):
#     """
#     Implements Faster R-CNN.
#
#     The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
#     image, and should be in 0-1 range. Different images can have different sizes.
#
#     The behavior of the model changes depending if it is in training or evaluation mode.
#
#     During training, the model expects both the input tensors, as well as a targets (list of dictionary),
#     containing:
#         - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
#           between 0 and H and 0 and W
#         - labels (Int64Tensor[N]): the class label for each ground-truth box
#
#     The model returns a Dict[Tensor] during training, containing the classification and regression
#     losses for both the RPN and the R-CNN.
#
#     During inference, the model requires only the input tensors, and returns the post-processed
#     predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
#     follows:
#         - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
#           0 and H and 0 and W
#         - labels (Int64Tensor[N]): the predicted labels for each image
#         - scores (Tensor[N]): the scores or each prediction
#
#     Arguments:
#         backbone (nn.Module): the network used to compute the features for the model.
#             It should contain a out_channels attribute, which indicates the number of output
#             channels that each feature map has (and it should be the same for all feature maps).
#             The backbone should return a single Tensor or and OrderedDict[Tensor].
#         num_classes (int): number of output classes of the model (including the background).
#             If box_predictor is specified, num_classes should be None.
#         min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
#         max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
#         image_mean (Tuple[float, float, float]): mean values used for input normalization.
#             They are generally the mean values of the dataset on which the backbone has been trained
#             on
#         image_std (Tuple[float, float, float]): std values used for input normalization.
#             They are generally the std values of the dataset on which the backbone has been trained on
#         rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
#             maps.
#         rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
#         rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
#         rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
#         rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
#         rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
#         rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
#         rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
#             considered as positive during training of the RPN.
#         rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
#             considered as negative during training of the RPN.
#         rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
#             for computing the loss
#         rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
#             of the RPN
#         rpn_score_thresh (float): during inference, only return proposals with a classification score
#             greater than rpn_score_thresh
#         box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
#             the locations indicated by the bounding boxes
#         box_head (nn.Module): module that takes the cropped feature maps as input
#         box_predictor (nn.Module): module that takes the output of box_head and returns the
#             classification logits and box regression deltas.
#         box_score_thresh (float): during inference, only return proposals with a classification score
#             greater than box_score_thresh
#         box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
#         box_detections_per_img (int): maximum number of detections per image, for all classes.
#         box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
#             considered as positive during training of the classification head
#         box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
#             considered as negative during training of the classification head
#         box_batch_size_per_image (int): number of proposals that are sampled during training of the
#             classification head
#         box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
#             of the classification head
#         bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
#             bounding boxes
#
#     """
#
#     def __init__(self, backbone, num_classes=None,
#                  # transform parameter
#                  min_size=800, max_size=1333,  # 预处理resize时限制的最小尺寸与最大尺寸
#                  image_mean=None, image_std=None,  # 预处理normalize时使用的均值和方差
#                  # RPN parameters
#                  rpn_anchor_generator=None, rpn_head=None,
#                  rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,  # rpn中在nms处理前保留的proposal数(根据score)
#                  rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn中在nms处理后保留的proposal数
#                  rpn_nms_thresh=0.7,  # rpn中进行nms处理时使用的iou阈值
#                  rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,  # rpn计算损失时，采集正负样本设置的阈值
#                  rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn计算损失时采样的样本数，以及正样本占总样本的比例
#                  rpn_score_thresh=0.0,
#                  # Box parameters
#                  box_roi_pool=None, box_head=None, box_predictor=None,
#                  # 移除低目标概率      fast rcnn中进行nms处理的阈值   对预测结果根据score排序取前100个目标
#                  box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
#                  box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,  # fast rcnn计算误差时，采集正负样本设置的阈值
#                  box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
#                  bbox_reg_weights=None):
#         if not hasattr(backbone, "out_channels"):
#             raise ValueError(
#                 "backbone should contain an attribute out_channels"
#                 "specifying the number of output channels  (assumed to be the"
#                 "same for all the levels"
#             )
#
#         assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
#         assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))
#
#         if num_classes is not None:
#             if box_predictor is not None:
#                 raise ValueError("num_classes should be None when box_predictor "
#                                  "is specified")
#         else:
#             if box_predictor is None:
#                 raise ValueError("num_classes should not be None when box_predictor "
#                                  "is not specified")
#
#         # 预测特征层的channels
#         out_channels = backbone.out_channels
#
#         # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
#         if rpn_anchor_generator is None:
#             anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
#             aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
#             rpn_anchor_generator = AnchorsGenerator(
#                 anchor_sizes, aspect_ratios
#             )
#
#         # 生成RPN通过滑动窗口预测网络部分
#         if rpn_head is None:
#             rpn_head = RPNHead(
#                 out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
#             )
#
#         # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
#         # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
#         rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
#         rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
#
#         # 定义整个RPN框架
#         rpn = RegionProposalNetwork(
#             rpn_anchor_generator, rpn_head,
#             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
#             rpn_batch_size_per_image, rpn_positive_fraction,
#             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
#             score_thresh=rpn_score_thresh)
#
#         #  Multi-scale RoIAlign pooling
#         if box_roi_pool is None:
#             box_roi_pool = MultiScaleRoIAlign(
#                 featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
#                 output_size=[7, 7],
#                 sampling_ratio=2)
#
#         # fast RCNN中roi pooling后的展平处理两个全连接层部分
#         if box_head is None:
#             resolution = box_roi_pool.output_size[0]  # 默认等于7
#             representation_size = 1024
#             box_head = TwoMLPHead(
#                 out_channels * resolution ** 2,
#                 representation_size
#             )
#
#         # 在box_head的输出上预测部分
#         if box_predictor is None:
#             representation_size = 1024
#             box_predictor = FastRCNNPredictor(
#                 representation_size,
#                 num_classes)
#
#         # 将roi pooling, box_head以及box_predictor结合在一起
#         roi_heads = RoIHeads(
#             # box
#             box_roi_pool, box_head, box_predictor,
#             box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
#             box_batch_size_per_image, box_positive_fraction,  # 512  0.25
#             bbox_reg_weights,
#             box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100
#
#         if image_mean is None:
#             image_mean = [0.485, 0.456, 0.406]
#         if image_std is None:
#             image_std = [0.229, 0.224, 0.225]
#
#         # 对数据进行标准化，缩放，打包成batch等处理部分
#         transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
#
#         super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)
#
#
# class FastRCNNPredictor(nn.Module):
#     """
#     Standard classification + bounding box regression layers
#     for Fast R-CNN.
#
#     Arguments:
#         in_channels (int): number of input channels
#         num_classes (int): number of output classes (including background)
#     """
#
#     def __init__(self, in_channels, num_classes):
#         super(FastRCNNPredictor, self).__init__()
#         self.cls_score = nn.Linear(in_channels, num_classes)
#         self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
#
#     def forward(self, x):
#         if x.dim() == 4:
#             assert list(x.shape[2:]) == [1, 1]
#         x = x.flatten(start_dim=1)
#         scores = self.cls_score(x)
#         bbox_deltas = self.bbox_pred(x)
#
#         return scores, bbox_deltas
#
#
# def resnet50_fpn_backbone(pretrain_path="", norm_layer=FrozenBatchNorm2d, trainable_layers=3, returned_layers=None, extra_blocks=None):
#     """
#     搭建resnet50_fpn——backbone
#     Args:
#         pretrain_path: resnet50的预训练权重，如果不使用就默认为空
#         norm_layer: 官方默认的是FrozenBatchNorm2d，即不会更新参数的bn层(因为如果batch_size设置的很小会导致效果更差，还不如不用bn层)
#                     如果自己的GPU显存很大可以设置很大的batch_size，那么自己可以传入正常的BatchNorm2d层
#                     (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
#         trainable_layers: 指定训练哪些层结构
#         returned_layers: 指定哪些层的输出需要返回
#         extra_blocks: 在输出的特征层基础上额外添加的层结构
#
#     Returns:
#
#     """
#     resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3],
#                              include_top=False,
#                              norm_layer=norm_layer)
#
#     if isinstance(norm_layer, FrozenBatchNorm2d):
#         overwrite_eps(resnet_backbone, 0.0)
#
#     if pretrain_path != "":
#         assert os.path.exists(pretrain_path), "{} is not exist.".format(pretrain_path)
#         # 载入预训练权重
#         print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))
#
#     # select layers that wont be frozen
#     assert 0 <= trainable_layers <= 5
#     layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
#
#     # 如果要训练所有层结构的话，不要忘了conv1后还有一个bn1
#     if trainable_layers == 5:
#         layers_to_train.append("bn1")
#
#     # freeze layers
#     for name, parameter in resnet_backbone.named_parameters():
#         # 只训练不在layers_to_train列表中的层结构
#         if all([not name.startswith(layer) for layer in layers_to_train]):
#             parameter.requires_grad_(False)
#
#     if extra_blocks is None:
#         extra_blocks = LastLevelMaxPool()
#
#     if returned_layers is None:
#         returned_layers = [1, 2, 3, 4]
#     # 返回的特征层个数肯定大于0小于5
#     assert min(returned_layers) > 0 and max(returned_layers) < 5
#
#     # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
#     return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
#
#     # in_channel 为layer4的输出特征矩阵channel = 2048
#     in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
#     # 记录resnet50提供给fpn的每个特征层channel
#     in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
#     # 通过fpn后得到的每个特征层的channel
#     out_channels = 256
#     return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
#
#
# def create_model(num_classes, device):
#     # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
#     # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
#     # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
#     # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
#     backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
#                                      trainable_layers=3)
#     # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
#     model = FasterRCNN(backbone=backbone, num_classes=91)
#     # 载入预训练模型权重
#     # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
#     weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location=device)
#     missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
#     if len(missing_keys) != 0 or len(unexpected_keys) != 0:
#         print("missing_keys: ", missing_keys)
#         print("unexpected_keys: ", unexpected_keys)
#
#     # get number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     # replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#
#     return model


if __name__ == "__main__":
    annotations_file = "K:\\LiHang\\Cell Instance Segmentation\\train_3.csv"
    img_root = "K:\\LiHang\\Cell Instance Segmentation\\train"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = DataSet(annotations_file, img_root, img_shape=(520, 704))

    # 返回 index, items
    # 其中 index 是索引，items 包含了[imgs, target]
    # imgs: [batch_size, img_size, img_size, 3]
    # target: [label, mask]，这里的信息可以在 class DataSet(Dataset) 里改变
    train_dataloader = DataLoader(train_set, batch_size=2)

    model = create_model(80, "cuda")
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)

    train_loss = []
    learning_rate = []
    val_map = []

