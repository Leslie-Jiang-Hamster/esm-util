import esm_util

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
dummy_data = [
    ("protein1", "M[15.99]KTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein3",  "K A <mask> I S Q"),
]

if __name__ == "__main__":
    representations = esm_util.ESMUtil().load_data(dummy_data).get_sequence_representations()
    print(representations[0])
    print(representations[0].shape)