import pandas as pd

item_d1_df = pd.read_csv("data/competition/item_domain_1.csv")
item_d2_df = pd.read_csv("data/competition/item_domain_2.csv")
train_df = pd.read_csv("data/competition/train.csv")
private_df = pd.read_csv("data/competition/private_test.csv")
submission_df = pd.read_csv("data/competition/submission_sample.csv")
print()

books_df = pd.read_csv("data/mid/Books.csv")
cds_and_vinyl_df = pd.read_csv("data/mid/CDs_and_Vinyl.csv")
movies_and_tv_df = pd.read_csv("data/mid/Movies_and_TV.csv")

print()

test_df = pd.read_csv("data/ready/_2_8/tgt_CDs_and_Vinyl_src_Books/test.csv")
train_meta_df = pd.read_csv(
    "data/ready/_2_8/tgt_CDs_and_Vinyl_src_Books/train_meta.csv", header=None
)
train_src_df = pd.read_csv(
    "data/ready/_2_8/tgt_CDs_and_Vinyl_src_Books/train_src.csv", header=None
)
train_tgt_df = pd.read_csv(
    "data/ready/_2_8/tgt_CDs_and_Vinyl_src_Books/train_tgt.csv", header=None
)

print()
