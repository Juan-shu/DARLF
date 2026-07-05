from __future__ import annotations

NOISY_GENE_SYMBOLS = {
    "CLOCK",
    "ARNTL",
    "BMAL1",
    "CRY1",
    "CRY2",
    "PER1",
    "PER2",
    "PER3",
    "DBP",
    "TEF",
    "NFIL3",
    "ISG15",
    "STAT1",
    "STAT2",
    "IRF7",
    "IRF9",
    "MX1",
    "MX2",
    "DDX58",
    "IFIH1",
    "HSPA1A",
    "HSPA1B",
    "HSPB1",
    "FOS",
    "FOSB",
    "JUN",
    "JUNB",
    "JUND",
    "EGR1",
    "EGR2",
    "EGR3",
    "NR4A1",
    "NR4A2",
    "NR4A3",
}

NOISY_GENE_PREFIXES = (
    "IFI",
    "IFIT",
    "OAS",
    "GBP",
    "HSP",
    "DNAJ",
    "ATF3",
)

AGING_PATHWAYS = {
    "telomere_maintenance": [
        "TERT", "TERC", "DKC1", "TERF1", "TERF2", "POT1", "ACD", "RTEL1",
    ],
    "dna_damage_repair": [
        "TP53", "ATM", "ATR", "CHEK1", "CHEK2", "BRCA1", "BRCA2",
        "RAD50", "RAD51", "XRCC5", "XRCC6", "PARP1", "WRN", "BLM",
    ],
    "mitochondrial_metabolism": [
        "PPARGC1A", "TFAM", "POLG", "SOD2", "MFN1", "MFN2", "OPA1",
        "SIRT3", "NDUFS1", "NDUFS2", "SDHA", "SDHB", "UQCRC1", "COX5A",
        "ATP5A1", "ATP5B",
    ],
    "autophagy_mtor": [
        "MTOR", "RPTOR", "RICTOR", "AKT1", "TSC1", "TSC2", "ULK1",
        "BECN1", "ATG5", "ATG7", "ATG12", "SQSTM1", "LAMP1", "LAMP2",
        "TFEB",
    ],
    "proteostasis": [
        "HSP90AA1", "HSPA8", "PSMA1", "PSMB5", "PSMC4", "PSMD1",
        "UBC", "VCP", "BAG3", "HSPA5", "CALR", "CANX", "DERL1",
        "UBE2D1", "UBE2D2",
    ],
    "senescence_cell_cycle": [
        "CDKN1A", "CDKN2A", "RB1", "CCND1", "LMNB1", "GLB1",
        "SERPINE1", "GDF15", "TP53", "CDK4", "CDK6", "E2F1",
    ],
    "inflammaging_core": [
        "IL1B", "IL6", "TNF", "CCL2", "NFKB1", "RELA", "TLR4",
        "STAT3", "TGFB1", "IL18", "NLRP3", "CASP1", "HMGB1",
        "MYD88", "TICAM1",
    ],
    "chromatin_epigenetic": [
        "DNMT1", "DNMT3A", "TET2", "EZH2", "SUZ12", "KDM6A",
        "EP300", "HDAC1", "HDAC2", "HDAC3", "SIRT1", "SIRT6",
        "KAT2A", "KAT2B",
    ],
    "oxidative_stress": [
        "CAT", "GPX1", "GPX4", "PRDX1", "PRDX2", "TXN", "TXNRD1",
        "NFE2L2", "KEAP1", "HMOX1", "NQO1", "GCLC", "GCLM",
    ],
    "growth_factor_insulin": [
        "IGF1", "IGF1R", "INSR", "IRS1", "IRS2", "FOXO1", "FOXO3",
        "FOXO4", "PIK3CA", "PIK3CB", "AKT2",
    ],
    "apoptosis_survival": [
        "BCL2", "BCL2L1", "BAX", "BAK1", "BAD", "BID", "CASP3",
        "CASP8", "CASP9", "CYCS", "APAF1", "XIAP", "BIRC5",
    ],
}

CELL_MARKERS = {
    "cd4_t": ["IL7R", "LTB", "MALAT1", "MAL", "LDHB", "CD4"],
    "cd8_t": ["NKG7", "CCL5", "GZMK", "TRAC", "CD3D", "CD8A", "CD8B"],
    "nk": ["NKG7", "GNLY", "KLRD1", "PRF1", "FCGR3A", "TYROBP", "NCAM1"],
    "b_cell": ["MS4A1", "CD79A", "CD79B", "HLA-DRA", "CD37", "CD19"],
    "monocyte": [
        "LYZ", "S100A8", "S100A9", "CTSS", "FCN1", "LST1", "CD14", "CD68",
    ],
    "dendritic": [
        "FCER1A", "CST3", "HLA-DPA1", "HLA-DPB1", "CLEC10A", "CLEC4C",
    ],
    "platelet": ["PPBP", "PF4", "NRGN", "GNG11", "SDPR", "GP9"],
    "endothelial": [
        "PECAM1", "CDH5", "VWF", "ENG", "CLDN5", "FLT1", "KDR",
    ],
    "fibroblast": [
        "COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "ACTA2",
    ],
    "neutrophil": [
        "ELANE", "MPO", "CEACAM8", "CXCR2", "FCGR3B", "ITGAM",
    ],
}

TISSUE_PROGRAM_BIASES = {
    "blood": ["inflammaging_core", "telomere_maintenance"],
    "brain": ["chromatin_epigenetic", "proteostasis", "autophagy_mtor"],
    "heart": ["mitochondrial_metabolism", "oxidative_stress"],
    "eye": ["oxidative_stress", "autophagy_mtor", "senescence_cell_cycle"],
    "liver": ["mitochondrial_metabolism", "growth_factor_insulin"],
    "adipose": ["growth_factor_insulin", "inflammaging_core"],
    "skin": ["senescence_cell_cycle", "dna_damage_repair"],
    "lung": ["oxidative_stress", "inflammaging_core"],
    "ovary": ["dna_damage_repair", "chromatin_epigenetic"],
}
