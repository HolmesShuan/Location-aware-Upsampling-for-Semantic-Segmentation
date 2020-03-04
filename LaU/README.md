# Location-aware-Upsampling
Pytorch implementation of "Location-aware Upsampling for Semantic Segmentation" (LaU). Pre-trained models, training and testing codes will be released in the next few months. [[arXiv Link]](https://arxiv.org/abs/1911.05250)

### Dependencies :
* **Python 3.5.6**
* **PyTorch 1.0.0**
* **GCC 7.3.0**

### Build :
```bash
cd LaU
bash make.sh
# python test.py
```

### Usage :
```python
from .LaU import LaU, LdU, LdU_MultiOutput

class DiffBiUpsampling(nn.Module):
    def __init__(self, k, category, offset_branch_input_channels, bottleneck_channel, batch_size, input_height, input_width, **kwargs):
        super(DiffBiUpsampling, self).__init__()

        self.k = k
        self.infer_w = nn.Sequential(
                            nn.Conv2d(offset_branch_input_channels, bottleneck_channel, 1, padding=0, bias=False, **kwargs),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(bottleneck_channel, self.k*self.k, 3, padding=1, bias=False, **kwargs)
                        )
        self.infer_h = nn.Sequential(
                            nn.Conv2d(offset_branch_input_channels, bottleneck_channel, 1, padding=0, bias=False, **kwargs),
                            nn.LeakyReLU(inplace=True),
                            nn.Conv2d(bottleneck_channel, self.k*self.k, 3, padding=1, bias=False, **kwargs)
                        )

        self.pixelshuffle = nn.PixelShuffle(self.k)

        nn.init.xavier_uniform_(self.infer_w[0].weight)
        nn.init.xavier_uniform_(self.infer_h[0].weight)
        nn.init.xavier_uniform_(self.infer_w[2].weight)
        nn.init.xavier_uniform_(self.infer_h[2].weight)

        self.lau = LaU(self.k, self.k).cuda()
    
    def forward(self, x, offset_branch_input):
        offsets_h = self.infer_h(offset_branch_input)
        offsets_w = self.infer_w(offset_branch_input)

        offsets_h = self.pixelshuffle(offsets_h)
        offsets_w = self.pixelshuffle(offsets_w)

        if self.training:
            offsets_return = torch.cat((offsets_h, offsets_w), dim=1) # (b, 2c, H, W)
        else:
            offsets_return = None

        offsets_h = offsets_h.repeat(1, x.size(1), 1, 1)
        offsets_w = offsets_w.repeat(1, x.size(1), 1, 1)

        y_offset = self.lau(x, offsets_h, offsets_w)
        
        return y_offset, offsets_return
```

### Acknowledgement :

We would like to thank [Deformable-Convolution-V2-PyTorch](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) and [pytorch-deform-conv](https://github.com/oeway/pytorch-deform-conv) for sharing their codes!

### Cite : 
```bib
@misc{he2019locationaware,
    title={Location-aware Upsampling for Semantic Segmentation},
    author={Xiangyu He and Zitao Mo and Qiang Chen and Anda Cheng and Peisong Wang and Jian Cheng},
    year={2019},
    eprint={1911.05250},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
