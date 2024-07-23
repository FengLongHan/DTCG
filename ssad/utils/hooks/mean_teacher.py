from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from mmrotate.utils import get_root_logger


@HOOKS.register_module()
class MeanTeacher(Hook):
    def __init__(
        self,
        momentum=0.9996,
        interval=1,
        # static_interval=3200,
        warm_up=100,
        start_steps=10000,
        skip_buffer=True,
    ):
        assert momentum >= 0 and momentum <= 1
        self.momentum = momentum
        assert isinstance(interval, int) and interval > 0
        # momentum warm up is disabled
        self.warm_up = warm_up
        self.interval = interval
        self.start_steps = start_steps
        self.skip_buffer = skip_buffer

        self.static_interval = 6400

    def before_run(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        assert hasattr(model, "teacher")
        assert hasattr(model, "student")
        assert hasattr(model, "static_teacher")

    def after_train_iter(self, runner):
        model = runner.model
        if is_module_wrapper(model):
            model = model.module
        curr_step = model.iter_count
        if curr_step % self.interval != 0 or curr_step < self.start_steps:
            return
        if curr_step == self.start_steps:
            logger = get_root_logger()
            logger.info(f"Start EMA Update at step {curr_step}")
            self.burnin_momentum_update(model, 0)
            # self.momentum_static_update(model, 0)

        else:
            self.momentum_update(model, self.momentum)

        if curr_step % self.static_interval == 0 & curr_step > self.static_interval:

            self.momentum_static_update(model, self.momentum)

    def burnin_momentum_update(self, model, momentum):
        if self.skip_buffer:
            for (
                (src_name, src_parm),
                (tgt_name, tgt_parm),
                (stgt_name, stgt_parm),
            ) in zip(
                model.student.named_parameters(),
                model.teacher.named_parameters(),
                model.static_teacher.named_parameters(),
            ):
                tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
                stgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
        else:
            for src_parm, dst_parm in zip(
                model.student.state_dict().values(),
                model.teacher.state_dict().values(),
                model.static_teacher.state_dict().values(),
            ):
                # exclude num_tracking
                if dst_parm.dtype.is_floating_point:
                    dst_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)

    def momentum_update(self, model, momentum):
        if self.skip_buffer:
            for (
                (src_name, src_parm),
                (tgt_name, tgt_parm),
                (stgt_name, stgt_parm),
            ) in zip(
                model.student.named_parameters(),
                model.teacher.named_parameters(),
                model.static_teacher.named_parameters(),
            ):
                tgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
                # stgt_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)
        else:
            for src_parm, dst_parm in zip(
                model.student.state_dict().values(),
                model.teacher.state_dict().values(),
                #   model.static_teacher.state_dict().values()
            ):
                # exclude num_tracking
                if dst_parm.dtype.is_floating_point:
                    dst_parm.data.mul_(momentum).add_(src_parm.data, alpha=1 - momentum)

    def momentum_static_update(self, model, momentum):

        if self.skip_buffer:
            for (
                (src_name, src_parm),
                (tgt_name, tgt_parm),
                (stgt_name, stgt_parm),
            ) in zip(
                model.student.named_parameters(),
                model.teacher.named_parameters(),
                model.static_teacher.named_parameters(),
            ):

                temp_parm = src_parm.data.clone()
                src_parm.data.copy_(stgt_parm.data)
                stgt_parm.data.copy_(temp_parm)

                stgt_parm.data.copy_(src_parm.data)
        else:
            for src_parm, dst_parm in zip(
                model.student.state_dict().values(),
                #   model.teacher.state_dict().values(),
                model.static_teacher.state_dict().values(),
            ):
                # exclude num_tracking
                if dst_parm.dtype.is_floating_point:

                    temp_param = src_parm.data.clone()
                    src_parm.data.copy_(dst_parm.data)
                    dst_parm.data.copy_(temp_param)
