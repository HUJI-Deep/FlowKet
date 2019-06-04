from .base_sampler import Sampler


class ToPlusMinusOneDecorator(Sampler):
    def __init__(self, sampler,):
        super(ToPlusMinusOneDecorator, self).__init__(input_size=sampler.input_size,
                                                      batch_size=sampler.batch_size,
                                                      mini_batch_size=sampler.mini_batch_size)
        self.sampler = sampler

    def __next__(self):
        batch = next(self.sampler)
        return 2 * batch - 1
