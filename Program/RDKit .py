from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

# 从 .mol 文件中读取化学结构
mol_file = 'mol1.mol'  # 替换为分子文件名
mol = Chem.MolFromMolFile(mol_file)

# 检查是否成功读取
if mol is not None:
    # 计算描述符
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    descriptors = {name: Descriptors.__dict__[name](mol) for name in descriptor_names}

    # 将描述符转换为 DataFrame
    descriptor_df = pd.DataFrame([descriptors])

    # 定义缺失的 SMARTS 表达式
    smarts_patterns = {

        'Aliphatic_Hydroxyl': "[C;!R][OH]",  # 链状羟基
        'Aliphatic_Hydroxyl_noTert': "[C;!R;!$(C([CH3])([CH3]))][OH]",  # 不包括叔OH的链状羟基
        'Aromatic_N_Functional_Groups': "cN",  # 连接到芳烃的 N
        'Aromatic_Carboxylic_Acids': "cC(=O)O",  # 芳香羧酸
        'Aromatic_Nitrogens': "n",  # 芳香氮
        'Aromatic_Amines': "cN",  # 芳香胺
        'Aromatic_Hydroxyl': "cO",  # 芳香羟基
        'Carbonyl_O': "C=O",  # 羰基
        'Carbonyl_O_noCOO': "[C;!$(C(=O)O)](=O)",  # 羰基 O（不包括 COOH）
        'Thioester': "C(=S)O",  # 硫代羰基
        'Imine': "C=N",  # 亚胺
        'Tertiary_Amines': "N(C)(C)(C)",  # 三级胺
        'Secondary_Amines': "N(C)(C)",  # 二级胺
        'Primary_Amines': "N",  # 一级胺
        'Hydroxylamine': "N(O)",  # 羟胺
        'Aldehydes': "[CH]=O",  # 醛
        'Alkyl_Carbamate': "C(=O)O[C,N]",  # 烷基碳酸酯
        'Alkyl_Halides': "C[Cl,Br,I,F]",  # 烷基卤化物
        'Allylic_Oxidation': "C=CC",  # 烯丙基氧化位点
        'Amides': "C(=O)N",  # 酰胺
        'Amines': "N",  # 脒基团
        'Anilines': "c1ccccc1N",  # 苯胺
        'Aryl_Methyl': "cC",  # 芳基甲基位点
        'Azides': "N=[N+]=[N-]",  # 叠氮化物
        'Azo': "N=N",  # 偶氮
        'Barbiturates': "c1ccccc1NC(=O)NC(=O)N",  # 巴比妥类
        'Benzene_Rings': "c1ccccc1",  # 苯环
        'Benzodiazepines': "c1ccccc1N",  # 苯二氮卓类
        'Bicyclic': "C1CC2CC1C2",  # 双环结构
        'Dihydropyridines': "C1=CC=CC=N1",  # 二氢吡啶
        'Epoxides': "C1OC1",  # 环氧化物
        'Esters': "C(=O)O",  # 酯
        'Ethers': "C-O-C",  # 醚
        'Furan': "c1ccoc1",  # 呋喃
        'Guanidines': "C(=N)N",  # 胍
        'Halogens': "[Cl,Br,I,F]",  # 卤素
        'Hydrazines': "N-N",  # 肼
        'Hydrazones': "C=N-N",  # 腙
        'Imidazoles': "c1ncnc1",  # 咪唑
        'Imides': "C(=O)NC(=O)",  # 酰亚胺
        'Isocyanates': "N=C=O",  # 异氰酸酯
        'Isothiocyanates': "N=C=S",  # 异硫氰酸盐
        'Ketones': "C(=O)",  # 酮
        'Lactams': "C(=O)N",  # β 内酰胺
        'Lactones': "O=C1OC[C@@H]1",  # 内酯
        'Methoxies': "CO",  # 甲氧基
        'Morpholines': "C1CCNCC1O",  # 吗啉
        'Nitriles': "C#N",  # 腈
        'Nitro': "N(=O)(=O)",  # 硝基
        'Nitro_Aromatic': "c[N+](=O)[O-]",  # 硝基芳香
        'Nitroso': "N=O",  # 亚硝基
        'Oxazoles': "c1ncnc1",  # 噁唑
        'Oximes': "C=N-O",  # 肟
        'Para_Hydroxylation': "c1ccc(O)c1",  # 对羟基化
        'Phosphonic_Acid': "P(=O)(O)O",  # 磷酸
        'Phosphonate_Ester': "P(=O)(O)O",  # 磷酸酯
        'Piperidines': "C1CCNCC1",  # 哌啶
        'Piperazines': "C1CCNCCN1",  # 胡椒片
        'Primary_Sulfonamides': "S(=O)(=O)N",  # 伯磺胺
        'Pyridines': "c1ccncc1",  # 吡啶
        'Quaternary_Nitrogens': "[N+](C)(C)(C)(C)",  # 季氮
        'Unbranched_Alkanes': "C-C-C",  # 无支链烷烃
        'Urea': "NC(=O)N",  # 脲键

    }

    # 用于保存结果的列表
    results = []

    for name, smarts in smarts_patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:  #
            matches = mol.GetSubstructMatches(pattern)
            results.append({'Fragment': name, 'Count': len(matches)})

    fragments_df = pd.DataFrame(results)

    fragments_df_pivot = fragments_df.set_index('Fragment').T

    final_df = pd.concat([descriptor_df.reset_index(drop=True), fragments_df_pivot.reset_index(drop=True)], axis=1)

    # 保存到 CSV 文件
    final_df.to_csv('mol1.csv', index=False)

    print("描述符和匹配结果已保存到文件中。")
else:
    print("无法读取化学结构文件")
