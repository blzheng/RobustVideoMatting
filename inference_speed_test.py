"""
python inference_speed_test.py \
    --model-variant mobilenetv3 \
    --resolution 1920 1080 \
    --downsample-ratio 0.25 \
    --precision float32
"""

import argparse
import torch
from tqdm import tqdm
import time
from utils_vis import make_dot, draw

from model.model import MattingNetwork

from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
import torch.profiler as profiler

# torch.backends.cudnn.benchmark = True
torch._C._jit_set_texpr_fuser_enabled(False)
torch.manual_seed(2020)

class InferenceSpeedTest:
    def __init__(self):
        self.parse_args()
        self.init_model()
        self.loop()
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-variant', type=str, required=True)
        parser.add_argument('--resolution', type=int, required=True, nargs=2)
        parser.add_argument('--downsample-ratio', type=float, required=True)
        parser.add_argument('--precision', type=str, default='float32')
        parser.add_argument('--disable-refiner', action='store_true')
        parser.add_argument('--mode', type=str, required=True)
        self.args = parser.parse_args()
        
    def init_model(self):
        self.device = 'cpu'
        self.precision = {'float32': torch.float32, 'float16': torch.float16}[self.args.precision]
        self.model = MattingNetwork(self.args.model_variant)
        self.model = self.model.to(device=self.device, dtype=self.precision).eval()
        if self.args.mode == "cpu":
            # pass
            self.model = torch.jit.script(self.model)
            self.model = torch.jit.freeze(self.model)
        elif self.args.mode == 'ipex_fp32':
            import intel_extension_for_pytorch as ipex
            #d = torch.randn(1, 3, 270, 480)
            #.to(memory_format=torch.channels_last)
            #self.model = self.model.to(memory_format=torch.channels_last)
            self.model.backbone =  self.model.backbone.to(memory_format=torch.channels_last)
            #self.model = ipex.optimize(self.model)
            self.model = torch.jit.script(self.model)
            self.model = torch.jit.freeze(self.model)

            #with torch.no_grad():
                #self.model.backbone = torch.jit.trace(self.model.backbone, d)
                #self.model.backbone = torch.jit.freeze(self.model.backbone)
                #y = self.model.backbone(d)
                #y = self.model.backbone(d)
                #graph = self.model.backbone.graph_for(d)
                #draw(graph).render('mobilenetv3_ipex_fp32')
        elif self.args.mode == 'ipex_int8':
            import intel_extension_for_pytorch as ipex
            from intel_extension_for_pytorch.quantization import prepare, convert
            d = torch.randn(1, 3, 270, 480)
            #.to(memory_format=torch.channels_last)
            self.model.backbone =  self.model.backbone.to(memory_format=torch.channels_last)
            qconfig = QConfig(activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_affine, dtype=torch.quint8),
                weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))
            prepared_model = prepare(self.model.backbone, qconfig, example_inputs=d, inplace=False)
            prepared_model(d)
            self.model.backbone = convert(prepared_model)

            with torch.no_grad():
                self.model.backbone = torch.jit.trace(self.model.backbone, d)
                self.model.aspp = torch.jit.script(self.model.aspp)
                self.model.decoder = torch.jit.script(self.model.decoder)
                self.model.project_mat = torch.jit.script(self.model.project_mat)
                self.model.refiner =  torch.jit.script(self.model.refiner)
                self.model.project_seg = torch.jit.script(self.model.project_seg)
                self.model.backbone = torch.jit.freeze(self.model.backbone)
                self.model.aspp = torch.jit.freeze(self.model.aspp)
                self.model.decoder = torch.jit.freeze(self.model.decoder)
                self.model.project_mat = torch.jit.freeze(self.model.project_mat)
                self.model.refiner =  torch.jit.freeze(self.model.refiner)
                self.model.project_seg = torch.jit.freeze(self.model.project_seg)
                y = self.model(d)
                y = self.model(d)
                #y = self.model.backbone(d)
                #y = self.model.backbone(d)
                #graph = self.model.backbone.graph_for(d)


                #draw(graph).render('mobilenetv3_ipex_int8')
    
    def loop(self):
        w, h = self.args.resolution
        src = torch.randn((1, 3, h, w), device=self.device, dtype=self.precision)
        #if self.args.mode != 'cpu':
        #    src = src.to(memory_format=torch.channels_last)
        with torch.no_grad():
            rec = None, None, None, None
            for _ in range(100):
                fgr, pha, *rec = self.model(src, *rec, self.args.downsample_ratio)
                #y = self.model(src, *rec, self.args.downsample_ratio)

            times = 200
            t_speed = 0
            for _ in range(times):
                t = time.time()
                fgr, pha, *rec = self.model(src, *rec, self.args.downsample_ratio)
                #y = self.model(src, *rec, self.args.downsample_ratio)
                t_speed += time.time() - t
            print('Speed: %.1f ms inference per %gx%g image' % (t_speed/times*1000, w, h))

            def trace_handler(prof):
                print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
                # prof.export_chrome_trace("rn50_trace_" + str(prof.step_num) + ".json")

            with profiler.profile(
                    activities=[profiler.ProfilerActivity.CPU],
                    schedule=torch.profiler.schedule(wait=10,warmup=50,active=10),
                    # son_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_result")
                    on_trace_ready=trace_handler) as p:
                for i in range(200):
                    y = self.model(src, *rec, self.args.downsample_ratio)
                    p.step()

if __name__ == '__main__':
    InferenceSpeedTest()
