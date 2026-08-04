import torch
import torch.nn as nn

try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    HAS_WEIGHTS = True
except ImportError:
    from torchvision.models import efficientnet_b0
    HAS_WEIGHTS = False


class BrainPreCheckModel(nn.Module):
    """
    Pre-Check Model berbasis EfficientNet-B0 (pretrained ImageNet) sebagai
    backbone ekstraksi fitur, dengan head klasifikasi biner (Valid vs Invalid).

    Nama atribut `backbone` dan `classifier` sengaja dipertahankan (sama seperti
    versi sebelumnya) agar kompatibel dengan train_precheck.py, yang membekukan
    `model.backbone` dan hanya melatih `model.classifier`.
    """

    def __init__(self):
        super(BrainPreCheckModel, self).__init__()

        # Backbone EfficientNet-B0. weights=None (TIDAK download ImageNet
        # pretrained) karena bobotnya akan langsung ditimpa total oleh
        # load_state_dict() dari checkpoint hasil training kita sendiri
        # (best_precheck_model.pth) di api.py. Download ImageNet di sini
        # cuma buang-buang bandwidth & RAM saat startup, dan jadi penyebab OOM.
        if HAS_WEIGHTS:
            self.backbone = efficientnet_b0(weights=None)
        else:
            self.backbone = efficientnet_b0(pretrained=False)

        # EfficientNet-B0 classifier bawaan: Sequential(Dropout, Linear(1280, 1000))
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()

        # Head klasifikasi biner: Valid (1) vs Invalid (0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 2),
        )

    def forward(self, x):
        feats = self.backbone(x)       # [B, 1280]
        return self.classifier(feats)  # [B, 2]


if __name__ == "__main__":
    # Uji coba apakah arsitektur model berhasil dimuat tanpa error
    model = BrainPreCheckModel()
    dummy_input = torch.randn(1, 3, 224, 224)  # Simulasi 1 gambar ukuran 224x224
    output = model(dummy_input)
    print(f"✨ Model Pre-Check (EfficientNet-B0) Sukses Dibuat! Ukuran Output: {output.shape}")
