class MLAConfig:
    def __init__(self, 
                 reduce_ratio_qk=2, reduce_ratio_vo=2, 
                 lora_rank=32, lora_alpha=64, 
                 path="./") -> None:
        self.reduce_ratio_qk = reduce_ratio_qk
        self.reduce_ratio_vo = reduce_ratio_vo
        self.r = lora_rank
        self.alpha = lora_alpha
        self.path = path
