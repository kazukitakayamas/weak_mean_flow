import copy
import torch


class EMA:
    def __init__(self, model, decay, period=1):
        if not 0 <= decay < 1 or period < 1:
            raise ValueError("EMAのdecayまたはperiodが不正です")

        self.net = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.period, self.num_updates = decay, period, 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1

        if self.num_updates % self.period:
            return

        d = self.decay ** self.period

        for pe, pm in zip(self.net.parameters(), model.parameters()):
            pe.copy_(
                (
                    pe.double()
                    + (1 - d) * (pm.double() - pe.double())
                ).float()
            )