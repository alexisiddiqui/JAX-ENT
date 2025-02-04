# Create a copy of loaded structure
create 4nx4, all
hide all
show cartoon

# Create bonds with thickness based on edge weights
distance bond_2_3, /4nx4//A/GLU`2/CA, /4nx4//A/CYS`3/CA
set dash_width, 1.0, bond_2_3
distance bond_2_7, /4nx4//A/GLU`2/CA, /4nx4//A/GLY`7/CA
set dash_width, 1.0, bond_2_7
distance bond_2_8, /4nx4//A/GLU`2/CA, /4nx4//A/TRP`8/CA
set dash_width, 2.0, bond_2_8
distance bond_2_10, /4nx4//A/GLU`2/CA, /4nx4//A/LEU`10/CA
set dash_width, 1.0, bond_2_10
distance bond_2_24, /4nx4//A/GLU`2/CA, /4nx4//A/THR`24/CA
set dash_width, 1.0, bond_2_24
distance bond_3_6, /4nx4//A/CYS`3/CA, /4nx4//A/CYS`6/CA
set dash_width, 1.0, bond_3_6
distance bond_3_7, /4nx4//A/CYS`3/CA, /4nx4//A/GLY`7/CA
set dash_width, 2.0, bond_3_7
distance bond_3_8, /4nx4//A/CYS`3/CA, /4nx4//A/TRP`8/CA
set dash_width, 7.0, bond_3_8
distance bond_3_24, /4nx4//A/CYS`3/CA, /4nx4//A/THR`24/CA
set dash_width, 2.0, bond_3_24
distance bond_4_5, /4nx4//A/ALA`4/CA, /4nx4//A/VAL`5/CA
set dash_width, 1.0, bond_4_5
distance bond_4_8, /4nx4//A/ALA`4/CA, /4nx4//A/TRP`8/CA
set dash_width, 1.0, bond_4_8
distance bond_4_24, /4nx4//A/ALA`4/CA, /4nx4//A/THR`24/CA
set dash_width, 6.0, bond_4_24
distance bond_5_6, /4nx4//A/VAL`5/CA, /4nx4//A/CYS`6/CA
set dash_width, 1.0, bond_5_6
distance bond_5_24, /4nx4//A/VAL`5/CA, /4nx4//A/THR`24/CA
set dash_width, 2.0, bond_5_24
distance bond_6_7, /4nx4//A/CYS`6/CA, /4nx4//A/GLY`7/CA
set dash_width, 1.0, bond_6_7
distance bond_7_8, /4nx4//A/GLY`7/CA, /4nx4//A/TRP`8/CA
set dash_width, 2.0, bond_7_8
distance bond_8_9, /4nx4//A/TRP`8/CA, /4nx4//A/ALA`9/CA
set dash_width, 1.0, bond_8_9
distance bond_9_10, /4nx4//A/ALA`9/CA, /4nx4//A/LEU`10/CA
set dash_width, 1.0, bond_9_10
distance bond_10_11, /4nx4//A/LEU`10/CA, /4nx4//A/PRO`11/CA
set dash_width, 1.0, bond_10_11
distance bond_12_13, /4nx4//A/HIS`12/CA, /4nx4//A/ASN`13/CA
set dash_width, 1.0, bond_12_13
distance bond_13_14, /4nx4//A/ASN`13/CA, /4nx4//A/ARG`14/CA
set dash_width, 1.0, bond_13_14
distance bond_13_15, /4nx4//A/ASN`13/CA, /4nx4//A/MET`15/CA
set dash_width, 1.0, bond_13_15
distance bond_14_15, /4nx4//A/ARG`14/CA, /4nx4//A/MET`15/CA
set dash_width, 2.0, bond_14_15
distance bond_15_16, /4nx4//A/MET`15/CA, /4nx4//A/GLN`16/CA
set dash_width, 1.0, bond_15_16
distance bond_15_25, /4nx4//A/MET`15/CA, /4nx4//A/ILE`25/CA
set dash_width, 2.0, bond_15_25
distance bond_16_17, /4nx4//A/GLN`16/CA, /4nx4//A/ALA`17/CA
set dash_width, 1.0, bond_16_17
distance bond_16_23, /4nx4//A/GLN`16/CA, /4nx4//A/CYS`23/CA
set dash_width, 1.0, bond_16_23
distance bond_16_24, /4nx4//A/GLN`16/CA, /4nx4//A/THR`24/CA
set dash_width, 1.0, bond_16_24
distance bond_16_25, /4nx4//A/GLN`16/CA, /4nx4//A/ILE`25/CA
set dash_width, 8.0, bond_16_25
distance bond_16_26, /4nx4//A/GLN`16/CA, /4nx4//A/CYS`26/CA
set dash_width, 1.0, bond_16_26
distance bond_17_18, /4nx4//A/ALA`17/CA, /4nx4//A/LEU`18/CA
set dash_width, 1.0, bond_17_18
distance bond_17_23, /4nx4//A/ALA`17/CA, /4nx4//A/CYS`23/CA
set dash_width, 2.0, bond_17_23
distance bond_17_25, /4nx4//A/ALA`17/CA, /4nx4//A/ILE`25/CA
set dash_width, 2.0, bond_17_25
distance bond_18_19, /4nx4//A/LEU`18/CA, /4nx4//A/THR`19/CA
set dash_width, 1.0, bond_18_19
distance bond_18_21, /4nx4//A/LEU`18/CA, /4nx4//A/CYS`21/CA
set dash_width, 2.0, bond_18_21
distance bond_18_23, /4nx4//A/LEU`18/CA, /4nx4//A/CYS`23/CA
set dash_width, 2.0, bond_18_23
distance bond_19_20, /4nx4//A/THR`19/CA, /4nx4//A/SER`20/CA
set dash_width, 1.0, bond_19_20
distance bond_19_21, /4nx4//A/THR`19/CA, /4nx4//A/CYS`21/CA
set dash_width, 2.0, bond_19_21
distance bond_20_21, /4nx4//A/SER`20/CA, /4nx4//A/CYS`21/CA
set dash_width, 2.0, bond_20_21
distance bond_20_53, /4nx4//A/SER`20/CA, /4nx4//A/ARG`53/CA
set dash_width, 1.0, bond_20_53
distance bond_21_53, /4nx4//A/CYS`21/CA, /4nx4//A/ARG`53/CA
set dash_width, 1.0, bond_21_53
distance bond_22_23, /4nx4//A/GLU`22/CA, /4nx4//A/CYS`23/CA
set dash_width, 2.0, bond_22_23
distance bond_23_24, /4nx4//A/CYS`23/CA, /4nx4//A/THR`24/CA
set dash_width, 1.0, bond_23_24
distance bond_24_25, /4nx4//A/THR`24/CA, /4nx4//A/ILE`25/CA
set dash_width, 1.0, bond_24_25
distance bond_25_26, /4nx4//A/ILE`25/CA, /4nx4//A/CYS`26/CA
set dash_width, 1.0, bond_25_26
distance bond_26_27, /4nx4//A/CYS`26/CA, /4nx4//A/PRO`27/CA
set dash_width, 1.0, bond_26_27
distance bond_28_29, /4nx4//A/ASP`28/CA, /4nx4//A/CYS`29/CA
set dash_width, 1.0, bond_28_29
distance bond_29_30, /4nx4//A/CYS`29/CA, /4nx4//A/PHE`30/CA
set dash_width, 1.0, bond_29_30
distance bond_30_31, /4nx4//A/PHE`30/CA, /4nx4//A/ARG`31/CA
set dash_width, 1.0, bond_30_31
distance bond_31_32, /4nx4//A/ARG`31/CA, /4nx4//A/GLN`32/CA
set dash_width, 1.0, bond_31_32
distance bond_31_76, /4nx4//A/ARG`31/CA, /4nx4//A/SER`76/CA
set dash_width, 1.0, bond_31_76
distance bond_32_33, /4nx4//A/GLN`32/CA, /4nx4//A/HIS`33/CA
set dash_width, 1.0, bond_32_33
distance bond_33_34, /4nx4//A/HIS`33/CA, /4nx4//A/PHE`34/CA
set dash_width, 1.0, bond_33_34
distance bond_34_35, /4nx4//A/PHE`34/CA, /4nx4//A/THR`35/CA
set dash_width, 1.0, bond_34_35
distance bond_35_36, /4nx4//A/THR`35/CA, /4nx4//A/ILE`36/CA
set dash_width, 1.0, bond_35_36
distance bond_36_37, /4nx4//A/ILE`36/CA, /4nx4//A/ALA`37/CA
set dash_width, 1.0, bond_36_37
distance bond_37_38, /4nx4//A/ALA`37/CA, /4nx4//A/LEU`38/CA
set dash_width, 1.0, bond_37_38
distance bond_37_41, /4nx4//A/ALA`37/CA, /4nx4//A/LYS`41/CA
set dash_width, 1.0, bond_37_41
distance bond_38_39, /4nx4//A/LEU`38/CA, /4nx4//A/LYS`39/CA
set dash_width, 1.0, bond_38_39
distance bond_38_41, /4nx4//A/LEU`38/CA, /4nx4//A/LYS`41/CA
set dash_width, 2.0, bond_38_41
distance bond_38_42, /4nx4//A/LEU`38/CA, /4nx4//A/HIS`42/CA
set dash_width, 1.0, bond_38_42
distance bond_39_40, /4nx4//A/LYS`39/CA, /4nx4//A/GLU`40/CA
set dash_width, 1.0, bond_39_40
distance bond_39_41, /4nx4//A/LYS`39/CA, /4nx4//A/LYS`41/CA
set dash_width, 2.0, bond_39_41
distance bond_40_41, /4nx4//A/GLU`40/CA, /4nx4//A/LYS`41/CA
set dash_width, 2.0, bond_40_41
distance bond_41_42, /4nx4//A/LYS`41/CA, /4nx4//A/HIS`42/CA
set dash_width, 1.0, bond_41_42
distance bond_42_43, /4nx4//A/HIS`42/CA, /4nx4//A/ILE`43/CA
set dash_width, 1.0, bond_42_43
distance bond_43_44, /4nx4//A/ILE`43/CA, /4nx4//A/THR`44/CA
set dash_width, 1.0, bond_43_44
distance bond_44_45, /4nx4//A/THR`44/CA, /4nx4//A/ASP`45/CA
set dash_width, 1.0, bond_44_45
distance bond_45_46, /4nx4//A/ASP`45/CA, /4nx4//A/MET`46/CA
set dash_width, 1.0, bond_45_46
distance bond_46_47, /4nx4//A/MET`46/CA, /4nx4//A/VAL`47/CA
set dash_width, 1.0, bond_46_47
distance bond_47_48, /4nx4//A/VAL`47/CA, /4nx4//A/CYS`48/CA
set dash_width, 1.0, bond_47_48
distance bond_47_52, /4nx4//A/VAL`47/CA, /4nx4//A/GLY`52/CA
set dash_width, 1.0, bond_47_52
distance bond_47_53, /4nx4//A/VAL`47/CA, /4nx4//A/ARG`53/CA
set dash_width, 1.0, bond_47_53
distance bond_47_54, /4nx4//A/VAL`47/CA, /4nx4//A/PRO`54/CA
set dash_width, 1.0, bond_47_54
distance bond_48_52, /4nx4//A/CYS`48/CA, /4nx4//A/GLY`52/CA
set dash_width, 2.0, bond_48_52
distance bond_48_53, /4nx4//A/CYS`48/CA, /4nx4//A/ARG`53/CA
set dash_width, 6.0, bond_48_53
distance bond_48_54, /4nx4//A/CYS`48/CA, /4nx4//A/PRO`54/CA
set dash_width, 1.0, bond_48_54
distance bond_50_51, /4nx4//A/ALA`50/CA, /4nx4//A/CYS`51/CA
set dash_width, 1.0, bond_50_51
distance bond_51_52, /4nx4//A/CYS`51/CA, /4nx4//A/GLY`52/CA
set dash_width, 1.0, bond_51_52
distance bond_52_53, /4nx4//A/GLY`52/CA, /4nx4//A/ARG`53/CA
set dash_width, 2.0, bond_52_53
distance bond_53_54, /4nx4//A/ARG`53/CA, /4nx4//A/PRO`54/CA
set dash_width, 2.0, bond_53_54
distance bond_56_57, /4nx4//A/LEU`56/CA, /4nx4//A/THR`57/CA
set dash_width, 1.0, bond_56_57
distance bond_56_58, /4nx4//A/LEU`56/CA, /4nx4//A/ASP`58/CA
set dash_width, 1.0, bond_56_58
distance bond_57_58, /4nx4//A/THR`57/CA, /4nx4//A/ASP`58/CA
set dash_width, 2.0, bond_57_58
distance bond_58_59, /4nx4//A/ASP`58/CA, /4nx4//A/ASP`59/CA
set dash_width, 1.0, bond_58_59
distance bond_59_60, /4nx4//A/ASP`59/CA, /4nx4//A/THR`60/CA
set dash_width, 1.0, bond_59_60
distance bond_60_61, /4nx4//A/THR`60/CA, /4nx4//A/GLN`61/CA
set dash_width, 1.0, bond_60_61
distance bond_61_62, /4nx4//A/GLN`61/CA, /4nx4//A/LEU`62/CA
set dash_width, 1.0, bond_61_62
distance bond_62_63, /4nx4//A/LEU`62/CA, /4nx4//A/LEU`63/CA
set dash_width, 1.0, bond_62_63
distance bond_63_64, /4nx4//A/LEU`63/CA, /4nx4//A/SER`64/CA
set dash_width, 1.0, bond_63_64
distance bond_64_65, /4nx4//A/SER`64/CA, /4nx4//A/TYR`65/CA
set dash_width, 1.0, bond_64_65
distance bond_65_66, /4nx4//A/TYR`65/CA, /4nx4//A/PHE`66/CA
set dash_width, 1.0, bond_65_66
distance bond_66_67, /4nx4//A/PHE`66/CA, /4nx4//A/SER`67/CA
set dash_width, 1.0, bond_66_67
distance bond_67_68, /4nx4//A/SER`67/CA, /4nx4//A/THR`68/CA
set dash_width, 1.0, bond_67_68
distance bond_68_69, /4nx4//A/THR`68/CA, /4nx4//A/LEU`69/CA
set dash_width, 1.0, bond_68_69
distance bond_69_70, /4nx4//A/LEU`69/CA, /4nx4//A/ASP`70/CA
set dash_width, 1.0, bond_69_70
distance bond_70_71, /4nx4//A/ASP`70/CA, /4nx4//A/ILE`71/CA
set dash_width, 1.0, bond_70_71
distance bond_71_72, /4nx4//A/ILE`71/CA, /4nx4//A/GLN`72/CA
set dash_width, 1.0, bond_71_72
distance bond_72_73, /4nx4//A/GLN`72/CA, /4nx4//A/LEU`73/CA
set dash_width, 1.0, bond_72_73
distance bond_73_74, /4nx4//A/LEU`73/CA, /4nx4//A/ARG`74/CA
set dash_width, 1.0, bond_73_74
distance bond_74_75, /4nx4//A/ARG`74/CA, /4nx4//A/GLU`75/CA
set dash_width, 1.0, bond_74_75
distance bond_74_77, /4nx4//A/ARG`74/CA, /4nx4//A/LEU`77/CA
set dash_width, 1.0, bond_74_77
distance bond_75_76, /4nx4//A/GLU`75/CA, /4nx4//A/SER`76/CA
set dash_width, 1.0, bond_75_76
distance bond_75_77, /4nx4//A/GLU`75/CA, /4nx4//A/LEU`77/CA
set dash_width, 1.0, bond_75_77
distance bond_76_77, /4nx4//A/SER`76/CA, /4nx4//A/LEU`77/CA
set dash_width, 2.0, bond_76_77
distance bond_77_78, /4nx4//A/LEU`77/CA, /4nx4//A/GLU`78/CA
set dash_width, 1.0, bond_77_78
distance bond_78_79, /4nx4//A/GLU`78/CA, /4nx4//A/PRO`79/CA
set dash_width, 1.0, bond_78_79
distance bond_80_81, /4nx4//A/ASP`80/CA, /4nx4//A/ALA`81/CA
set dash_width, 1.0, bond_80_81
distance bond_81_82, /4nx4//A/ALA`81/CA, /4nx4//A/TYR`82/CA
set dash_width, 1.0, bond_81_82
distance bond_82_83, /4nx4//A/TYR`82/CA, /4nx4//A/ALA`83/CA
set dash_width, 1.0, bond_82_83
distance bond_83_84, /4nx4//A/ALA`83/CA, /4nx4//A/LEU`84/CA
set dash_width, 1.0, bond_83_84
distance bond_84_85, /4nx4//A/LEU`84/CA, /4nx4//A/PHE`85/CA
set dash_width, 1.0, bond_84_85
distance bond_85_86, /4nx4//A/PHE`85/CA, /4nx4//A/HIS`86/CA
set dash_width, 1.0, bond_85_86
distance bond_86_87, /4nx4//A/HIS`86/CA, /4nx4//A/LYS`87/CA
set dash_width, 1.0, bond_86_87
distance bond_87_88, /4nx4//A/LYS`87/CA, /4nx4//A/LYS`88/CA
set dash_width, 1.0, bond_87_88
distance bond_88_89, /4nx4//A/LYS`88/CA, /4nx4//A/LEU`89/CA
set dash_width, 1.0, bond_88_89
distance bond_89_90, /4nx4//A/LEU`89/CA, /4nx4//A/THR`90/CA
set dash_width, 1.0, bond_89_90
distance bond_90_91, /4nx4//A/THR`90/CA, /4nx4//A/GLU`91/CA
set dash_width, 1.0, bond_90_91
distance bond_91_92, /4nx4//A/GLU`91/CA, /4nx4//A/GLY`92/CA
set dash_width, 1.0, bond_91_92
distance bond_92_93, /4nx4//A/GLY`92/CA, /4nx4//A/VAL`93/CA
set dash_width, 1.0, bond_92_93
distance bond_93_94, /4nx4//A/VAL`93/CA, /4nx4//A/LEU`94/CA
set dash_width, 1.0, bond_93_94
distance bond_94_95, /4nx4//A/LEU`94/CA, /4nx4//A/MET`95/CA
set dash_width, 1.0, bond_94_95
distance bond_95_96, /4nx4//A/MET`95/CA, /4nx4//A/ARG`96/CA
set dash_width, 1.0, bond_95_96
distance bond_95_97, /4nx4//A/MET`95/CA, /4nx4//A/ASP`97/CA
set dash_width, 1.0, bond_95_97
distance bond_96_97, /4nx4//A/ARG`96/CA, /4nx4//A/ASP`97/CA
set dash_width, 2.0, bond_96_97
distance bond_97_98, /4nx4//A/ASP`97/CA, /4nx4//A/PRO`98/CA
set dash_width, 1.0, bond_97_98
distance bond_99_100, /4nx4//A/LYS`99/CA, /4nx4//A/PHE`100/CA
set dash_width, 2.0, bond_99_100
distance bond_99_112, /4nx4//A/LYS`99/CA, /4nx4//A/TYR`112/CA
set dash_width, 1.0, bond_99_112
distance bond_100_101, /4nx4//A/PHE`100/CA, /4nx4//A/LEU`101/CA
set dash_width, 1.0, bond_100_101
distance bond_100_110, /4nx4//A/PHE`100/CA, /4nx4//A/PHE`110/CA
set dash_width, 2.0, bond_100_110
distance bond_100_112, /4nx4//A/PHE`100/CA, /4nx4//A/TYR`112/CA
set dash_width, 1.0, bond_100_112
distance bond_101_102, /4nx4//A/LEU`101/CA, /4nx4//A/TRP`102/CA
set dash_width, 1.0, bond_101_102
distance bond_101_109, /4nx4//A/LEU`101/CA, /4nx4//A/GLY`109/CA
set dash_width, 1.0, bond_101_109
distance bond_101_110, /4nx4//A/LEU`101/CA, /4nx4//A/PHE`110/CA
set dash_width, 7.0, bond_101_110
distance bond_101_111, /4nx4//A/LEU`101/CA, /4nx4//A/ILE`111/CA
set dash_width, 1.0, bond_101_111
distance bond_102_103, /4nx4//A/TRP`102/CA, /4nx4//A/CYS`103/CA
set dash_width, 1.0, bond_102_103
distance bond_102_108, /4nx4//A/TRP`102/CA, /4nx4//A/PHE`108/CA
set dash_width, 2.0, bond_102_108
distance bond_102_110, /4nx4//A/TRP`102/CA, /4nx4//A/PHE`110/CA
set dash_width, 1.0, bond_102_110
distance bond_103_107, /4nx4//A/CYS`103/CA, /4nx4//A/SER`107/CA
set dash_width, 1.0, bond_103_107
distance bond_103_108, /4nx4//A/CYS`103/CA, /4nx4//A/PHE`108/CA
set dash_width, 8.0, bond_103_108
distance bond_104_105, /4nx4//A/ALA`104/CA, /4nx4//A/GLN`105/CA
set dash_width, 1.0, bond_104_105
distance bond_104_108, /4nx4//A/ALA`104/CA, /4nx4//A/PHE`108/CA
set dash_width, 1.0, bond_104_108
distance bond_104_133, /4nx4//A/ALA`104/CA, /4nx4//A/LYS`133/CA
set dash_width, 1.0, bond_104_133
distance bond_105_106, /4nx4//A/GLN`105/CA, /4nx4//A/CYS`106/CA
set dash_width, 1.0, bond_105_106
distance bond_105_133, /4nx4//A/GLN`105/CA, /4nx4//A/LYS`133/CA
set dash_width, 1.0, bond_105_133
distance bond_106_107, /4nx4//A/CYS`106/CA, /4nx4//A/SER`107/CA
set dash_width, 1.0, bond_106_107
distance bond_106_108, /4nx4//A/CYS`106/CA, /4nx4//A/PHE`108/CA
set dash_width, 1.0, bond_106_108
distance bond_107_108, /4nx4//A/SER`107/CA, /4nx4//A/PHE`108/CA
set dash_width, 2.0, bond_107_108
distance bond_108_109, /4nx4//A/PHE`108/CA, /4nx4//A/GLY`109/CA
set dash_width, 1.0, bond_108_109
distance bond_109_110, /4nx4//A/GLY`109/CA, /4nx4//A/PHE`110/CA
set dash_width, 1.0, bond_109_110
distance bond_110_111, /4nx4//A/PHE`110/CA, /4nx4//A/ILE`111/CA
set dash_width, 1.0, bond_110_111
distance bond_111_112, /4nx4//A/ILE`111/CA, /4nx4//A/TYR`112/CA
set dash_width, 1.0, bond_111_112
distance bond_113_114, /4nx4//A/GLU`113/CA, /4nx4//A/ARG`114/CA
set dash_width, 2.0, bond_113_114
distance bond_114_116, /4nx4//A/ARG`114/CA, /4nx4//A/GLN`116/CA
set dash_width, 1.0, bond_114_116
distance bond_115_116, /4nx4//A/GLU`115/CA, /4nx4//A/GLN`116/CA
set dash_width, 2.0, bond_115_116
distance bond_116_117, /4nx4//A/GLN`116/CA, /4nx4//A/LEU`117/CA
set dash_width, 1.0, bond_116_117
distance bond_116_118, /4nx4//A/GLN`116/CA, /4nx4//A/GLU`118/CA
set dash_width, 1.0, bond_116_118
distance bond_117_118, /4nx4//A/LEU`117/CA, /4nx4//A/GLU`118/CA
set dash_width, 2.0, bond_117_118
distance bond_118_119, /4nx4//A/GLU`118/CA, /4nx4//A/ALA`119/CA
set dash_width, 1.0, bond_118_119
distance bond_118_128, /4nx4//A/GLU`118/CA, /4nx4//A/PHE`128/CA
set dash_width, 2.0, bond_118_128
distance bond_119_120, /4nx4//A/ALA`119/CA, /4nx4//A/THR`120/CA
set dash_width, 1.0, bond_119_120
distance bond_119_126, /4nx4//A/ALA`119/CA, /4nx4//A/GLN`126/CA
set dash_width, 1.0, bond_119_126
distance bond_119_128, /4nx4//A/ALA`119/CA, /4nx4//A/PHE`128/CA
set dash_width, 6.0, bond_119_128
distance bond_119_129, /4nx4//A/ALA`119/CA, /4nx4//A/CYS`129/CA
set dash_width, 1.0, bond_119_129
distance bond_120_121, /4nx4//A/THR`120/CA, /4nx4//A/CYS`121/CA
set dash_width, 1.0, bond_120_121
distance bond_120_126, /4nx4//A/THR`120/CA, /4nx4//A/GLN`126/CA
set dash_width, 2.0, bond_120_126
distance bond_120_128, /4nx4//A/THR`120/CA, /4nx4//A/PHE`128/CA
set dash_width, 1.0, bond_120_128
distance bond_121_125, /4nx4//A/CYS`121/CA, /4nx4//A/HIS`125/CA
set dash_width, 2.0, bond_121_125
distance bond_121_126, /4nx4//A/CYS`121/CA, /4nx4//A/GLN`126/CA
set dash_width, 7.0, bond_121_126
distance bond_123_124, /4nx4//A/GLN`123/CA, /4nx4//A/CYS`124/CA
set dash_width, 1.0, bond_123_124
distance bond_124_125, /4nx4//A/CYS`124/CA, /4nx4//A/HIS`125/CA
set dash_width, 1.0, bond_124_125
distance bond_125_126, /4nx4//A/HIS`125/CA, /4nx4//A/GLN`126/CA
set dash_width, 2.0, bond_125_126
distance bond_126_127, /4nx4//A/GLN`126/CA, /4nx4//A/THR`127/CA
set dash_width, 1.0, bond_126_127
distance bond_127_128, /4nx4//A/THR`127/CA, /4nx4//A/PHE`128/CA
set dash_width, 1.0, bond_127_128
distance bond_127_136, /4nx4//A/THR`127/CA, /4nx4//A/TRP`136/CA
set dash_width, 1.0, bond_127_136
distance bond_128_129, /4nx4//A/PHE`128/CA, /4nx4//A/CYS`129/CA
set dash_width, 1.0, bond_128_129
distance bond_128_134, /4nx4//A/PHE`128/CA, /4nx4//A/ARG`134/CA
set dash_width, 2.0, bond_128_134
distance bond_128_135, /4nx4//A/PHE`128/CA, /4nx4//A/GLN`135/CA
set dash_width, 1.0, bond_128_135
distance bond_128_136, /4nx4//A/PHE`128/CA, /4nx4//A/TRP`136/CA
set dash_width, 1.0, bond_128_136
distance bond_129_130, /4nx4//A/CYS`129/CA, /4nx4//A/VAL`130/CA
set dash_width, 1.0, bond_129_130
distance bond_129_133, /4nx4//A/CYS`129/CA, /4nx4//A/LYS`133/CA
set dash_width, 1.0, bond_129_133
distance bond_129_134, /4nx4//A/CYS`129/CA, /4nx4//A/ARG`134/CA
set dash_width, 8.0, bond_129_134
distance bond_129_135, /4nx4//A/CYS`129/CA, /4nx4//A/GLN`135/CA
set dash_width, 1.0, bond_129_135
distance bond_130_131, /4nx4//A/VAL`130/CA, /4nx4//A/ARG`131/CA
set dash_width, 1.0, bond_130_131
distance bond_130_134, /4nx4//A/VAL`130/CA, /4nx4//A/ARG`134/CA
set dash_width, 1.0, bond_130_134
distance bond_131_132, /4nx4//A/ARG`131/CA, /4nx4//A/CYS`132/CA
set dash_width, 1.0, bond_131_132
distance bond_132_133, /4nx4//A/CYS`132/CA, /4nx4//A/LYS`133/CA
set dash_width, 1.0, bond_132_133
distance bond_132_134, /4nx4//A/CYS`132/CA, /4nx4//A/ARG`134/CA
set dash_width, 1.0, bond_132_134
distance bond_133_134, /4nx4//A/LYS`133/CA, /4nx4//A/ARG`134/CA
set dash_width, 2.0, bond_133_134
distance bond_134_135, /4nx4//A/ARG`134/CA, /4nx4//A/GLN`135/CA
set dash_width, 1.0, bond_134_135
distance bond_135_136, /4nx4//A/GLN`135/CA, /4nx4//A/TRP`136/CA
set dash_width, 1.0, bond_135_136
distance bond_136_137, /4nx4//A/TRP`136/CA, /4nx4//A/GLU`137/CA
set dash_width, 1.0, bond_136_137
distance bond_137_138, /4nx4//A/GLU`137/CA, /4nx4//A/GLU`138/CA
set dash_width, 1.0, bond_137_138
distance bond_138_139, /4nx4//A/GLU`138/CA, /4nx4//A/GLN`139/CA
set dash_width, 1.0, bond_138_139
distance bond_139_140, /4nx4//A/GLN`139/CA, /4nx4//A/HIS`140/CA
set dash_width, 1.0, bond_139_140
distance bond_139_141, /4nx4//A/GLN`139/CA, /4nx4//A/ARG`141/CA
set dash_width, 1.0, bond_139_141
distance bond_140_141, /4nx4//A/HIS`140/CA, /4nx4//A/ARG`141/CA
set dash_width, 2.0, bond_140_141
distance bond_140_143, /4nx4//A/HIS`140/CA, /4nx4//A/ARG`143/CA
set dash_width, 1.0, bond_140_143
distance bond_141_142, /4nx4//A/ARG`141/CA, /4nx4//A/GLY`142/CA
set dash_width, 1.0, bond_141_142
distance bond_141_143, /4nx4//A/ARG`141/CA, /4nx4//A/ARG`143/CA
set dash_width, 1.0, bond_141_143
distance bond_142_143, /4nx4//A/GLY`142/CA, /4nx4//A/ARG`143/CA
set dash_width, 2.0, bond_142_143
distance bond_143_144, /4nx4//A/ARG`143/CA, /4nx4//A/SER`144/CA
set dash_width, 1.0, bond_143_144
distance bond_144_145, /4nx4//A/SER`144/CA, /4nx4//A/CYS`145/CA
set dash_width, 1.0, bond_144_145
distance bond_144_147, /4nx4//A/SER`144/CA, /4nx4//A/ASP`147/CA
set dash_width, 1.0, bond_144_147
distance bond_145_146, /4nx4//A/CYS`145/CA, /4nx4//A/GLU`146/CA
set dash_width, 1.0, bond_145_146
distance bond_146_147, /4nx4//A/GLU`146/CA, /4nx4//A/ASP`147/CA
set dash_width, 1.0, bond_146_147
distance bond_147_148, /4nx4//A/ASP`147/CA, /4nx4//A/PHE`148/CA
set dash_width, 1.0, bond_147_148
distance bond_148_149, /4nx4//A/PHE`148/CA, /4nx4//A/GLN`149/CA
set dash_width, 1.0, bond_148_149
distance bond_149_150, /4nx4//A/GLN`149/CA, /4nx4//A/ASN`150/CA
set dash_width, 1.0, bond_149_150
distance bond_150_151, /4nx4//A/ASN`150/CA, /4nx4//A/TRP`151/CA
set dash_width, 1.0, bond_150_151
distance bond_151_152, /4nx4//A/TRP`151/CA, /4nx4//A/LYS`152/CA
set dash_width, 1.0, bond_151_152
distance bond_152_153, /4nx4//A/LYS`152/CA, /4nx4//A/ARG`153/CA
set dash_width, 1.0, bond_152_153
distance bond_153_154, /4nx4//A/ARG`153/CA, /4nx4//A/MET`154/CA
set dash_width, 1.0, bond_153_154
distance bond_153_155, /4nx4//A/ARG`153/CA, /4nx4//A/ASN`155/CA
set dash_width, 1.0, bond_153_155
distance bond_154_155, /4nx4//A/MET`154/CA, /4nx4//A/ASN`155/CA
set dash_width, 1.0, bond_154_155
distance bond_154_156, /4nx4//A/MET`154/CA, /4nx4//A/ASP`156/CA
set dash_width, 1.0, bond_154_156
distance bond_155_156, /4nx4//A/ASN`155/CA, /4nx4//A/ASP`156/CA
set dash_width, 2.0, bond_155_156
distance bond_156_157, /4nx4//A/ASP`156/CA, /4nx4//A/PRO`157/CA
set dash_width, 1.0, bond_156_157
distance bond_158_159, /4nx4//A/GLU`158/CA, /4nx4//A/TYR`159/CA
set dash_width, 1.0, bond_158_159
distance bond_159_160, /4nx4//A/TYR`159/CA, /4nx4//A/GLN`160/CA
set dash_width, 1.0, bond_159_160
distance bond_160_161, /4nx4//A/GLN`160/CA, /4nx4//A/ALA`161/CA
set dash_width, 1.0, bond_160_161
distance bond_161_162, /4nx4//A/ALA`161/CA, /4nx4//A/GLN`162/CA
set dash_width, 1.0, bond_161_162
distance bond_162_163, /4nx4//A/GLN`162/CA, /4nx4//A/GLY`163/CA
set dash_width, 1.0, bond_162_163
distance bond_163_164, /4nx4//A/GLY`163/CA, /4nx4//A/LEU`164/CA
set dash_width, 1.0, bond_163_164
distance bond_164_165, /4nx4//A/LEU`164/CA, /4nx4//A/ALA`165/CA
set dash_width, 1.0, bond_164_165
distance bond_165_166, /4nx4//A/ALA`165/CA, /4nx4//A/MET`166/CA
set dash_width, 1.0, bond_165_166
distance bond_166_167, /4nx4//A/MET`166/CA, /4nx4//A/TYR`167/CA
set dash_width, 1.0, bond_166_167
distance bond_167_168, /4nx4//A/TYR`167/CA, /4nx4//A/LEU`168/CA
set dash_width, 1.0, bond_167_168
distance bond_168_169, /4nx4//A/LEU`168/CA, /4nx4//A/GLN`169/CA
set dash_width, 1.0, bond_168_169
distance bond_168_171, /4nx4//A/LEU`168/CA, /4nx4//A/ASN`171/CA
set dash_width, 1.0, bond_168_171
distance bond_169_170, /4nx4//A/GLN`169/CA, /4nx4//A/GLU`170/CA
set dash_width, 1.0, bond_169_170
distance bond_169_172, /4nx4//A/GLN`169/CA, /4nx4//A/GLY`172/CA
set dash_width, 1.0, bond_169_172
distance bond_170_171, /4nx4//A/GLU`170/CA, /4nx4//A/ASN`171/CA
set dash_width, 1.0, bond_170_171
distance bond_170_172, /4nx4//A/GLU`170/CA, /4nx4//A/GLY`172/CA
set dash_width, 1.0, bond_170_172
distance bond_171_172, /4nx4//A/ASN`171/CA, /4nx4//A/GLY`172/CA
set dash_width, 2.0, bond_171_172
distance bond_171_184, /4nx4//A/ASN`171/CA, /4nx4//A/LEU`184/CA
set dash_width, 1.0, bond_171_184
distance bond_172_173, /4nx4//A/GLY`172/CA, /4nx4//A/ILE`173/CA
set dash_width, 1.0, bond_172_173
distance bond_172_182, /4nx4//A/GLY`172/CA, /4nx4//A/TYR`182/CA
set dash_width, 2.0, bond_172_182
distance bond_172_184, /4nx4//A/GLY`172/CA, /4nx4//A/LEU`184/CA
set dash_width, 1.0, bond_172_184
distance bond_173_174, /4nx4//A/ILE`173/CA, /4nx4//A/ASP`174/CA
set dash_width, 1.0, bond_173_174
distance bond_173_182, /4nx4//A/ILE`173/CA, /4nx4//A/TYR`182/CA
set dash_width, 6.0, bond_173_182
distance bond_173_183, /4nx4//A/ILE`173/CA, /4nx4//A/ALA`183/CA
set dash_width, 1.0, bond_173_183
distance bond_174_175, /4nx4//A/ASP`174/CA, /4nx4//A/CYS`175/CA
set dash_width, 1.0, bond_174_175
distance bond_174_179, /4nx4//A/ASP`174/CA, /4nx4//A/LYS`179/CA
set dash_width, 1.0, bond_174_179
distance bond_174_180, /4nx4//A/ASP`174/CA, /4nx4//A/PHE`180/CA
set dash_width, 2.0, bond_174_180
distance bond_174_182, /4nx4//A/ASP`174/CA, /4nx4//A/TYR`182/CA
set dash_width, 1.0, bond_174_182
distance bond_175_179, /4nx4//A/CYS`175/CA, /4nx4//A/LYS`179/CA
set dash_width, 2.0, bond_175_179
distance bond_175_180, /4nx4//A/CYS`175/CA, /4nx4//A/PHE`180/CA
set dash_width, 7.0, bond_175_180
distance bond_177_178, /4nx4//A/LYS`177/CA, /4nx4//A/CYS`178/CA
set dash_width, 1.0, bond_177_178
distance bond_178_179, /4nx4//A/CYS`178/CA, /4nx4//A/LYS`179/CA
set dash_width, 1.0, bond_178_179
distance bond_179_180, /4nx4//A/LYS`179/CA, /4nx4//A/PHE`180/CA
set dash_width, 2.0, bond_179_180
distance bond_180_181, /4nx4//A/PHE`180/CA, /4nx4//A/SER`181/CA
set dash_width, 1.0, bond_180_181
distance bond_181_182, /4nx4//A/SER`181/CA, /4nx4//A/TYR`182/CA
set dash_width, 1.0, bond_181_182
distance bond_182_183, /4nx4//A/TYR`182/CA, /4nx4//A/ALA`183/CA
set dash_width, 1.0, bond_182_183
distance bond_183_184, /4nx4//A/ALA`183/CA, /4nx4//A/LEU`184/CA
set dash_width, 1.0, bond_183_184
distance bond_184_185, /4nx4//A/LEU`184/CA, /4nx4//A/ALA`185/CA
set dash_width, 1.0, bond_184_185
distance bond_185_186, /4nx4//A/ALA`185/CA, /4nx4//A/ARG`186/CA
set dash_width, 1.0, bond_185_186
distance bond_185_276, /4nx4//A/ALA`185/CA, /4nx4//A/ILE`276/CA
set dash_width, 1.0, bond_185_276
distance bond_186_276, /4nx4//A/ARG`186/CA, /4nx4//A/ILE`276/CA
set dash_width, 3.0, bond_186_276
distance bond_187_188, /4nx4//A/GLY`187/CA, /4nx4//A/GLY`188/CA
set dash_width, 2.0, bond_187_188
distance bond_187_276, /4nx4//A/GLY`187/CA, /4nx4//A/ILE`276/CA
set dash_width, 1.0, bond_187_276
distance bond_188_189, /4nx4//A/GLY`188/CA, /4nx4//A/CYS`189/CA
set dash_width, 2.0, bond_188_189
distance bond_189_191, /4nx4//A/CYS`189/CA, /4nx4//A/HIS`191/CA
set dash_width, 1.0, bond_189_191
distance bond_190_191, /4nx4//A/MET`190/CA, /4nx4//A/HIS`191/CA
set dash_width, 2.0, bond_190_191
distance bond_191_192, /4nx4//A/HIS`191/CA, /4nx4//A/PHE`192/CA
set dash_width, 1.0, bond_191_192
distance bond_191_201, /4nx4//A/HIS`191/CA, /4nx4//A/PHE`201/CA
set dash_width, 2.0, bond_191_201
distance bond_192_193, /4nx4//A/PHE`192/CA, /4nx4//A/HIS`193/CA
set dash_width, 1.0, bond_192_193
distance bond_192_199, /4nx4//A/PHE`192/CA, /4nx4//A/HIS`199/CA
set dash_width, 1.0, bond_192_199
distance bond_192_200, /4nx4//A/PHE`192/CA, /4nx4//A/GLN`200/CA
set dash_width, 1.0, bond_192_200
distance bond_192_201, /4nx4//A/PHE`192/CA, /4nx4//A/PHE`201/CA
set dash_width, 9.0, bond_192_201
distance bond_192_202, /4nx4//A/PHE`192/CA, /4nx4//A/CYS`202/CA
set dash_width, 1.0, bond_192_202
distance bond_193_194, /4nx4//A/HIS`193/CA, /4nx4//A/CYS`194/CA
set dash_width, 1.0, bond_193_194
distance bond_193_199, /4nx4//A/HIS`193/CA, /4nx4//A/HIS`199/CA
set dash_width, 2.0, bond_193_199
distance bond_193_201, /4nx4//A/HIS`193/CA, /4nx4//A/PHE`201/CA
set dash_width, 2.0, bond_193_201
distance bond_194_198, /4nx4//A/CYS`194/CA, /4nx4//A/ARG`198/CA
set dash_width, 2.0, bond_194_198
distance bond_194_199, /4nx4//A/CYS`194/CA, /4nx4//A/HIS`199/CA
set dash_width, 7.0, bond_194_199
distance bond_195_196, /4nx4//A/THR`195/CA, /4nx4//A/GLN`196/CA
set dash_width, 1.0, bond_195_196
distance bond_195_199, /4nx4//A/THR`195/CA, /4nx4//A/HIS`199/CA
set dash_width, 1.0, bond_195_199
distance bond_196_197, /4nx4//A/GLN`196/CA, /4nx4//A/CYS`197/CA
set dash_width, 1.0, bond_196_197
distance bond_197_198, /4nx4//A/CYS`197/CA, /4nx4//A/ARG`198/CA
set dash_width, 1.0, bond_197_198
distance bond_198_199, /4nx4//A/ARG`198/CA, /4nx4//A/HIS`199/CA
set dash_width, 2.0, bond_198_199
distance bond_199_200, /4nx4//A/HIS`199/CA, /4nx4//A/GLN`200/CA
set dash_width, 1.0, bond_199_200
distance bond_200_201, /4nx4//A/GLN`200/CA, /4nx4//A/PHE`201/CA
set dash_width, 1.0, bond_200_201
distance bond_200_209, /4nx4//A/GLN`200/CA, /4nx4//A/PHE`209/CA
set dash_width, 2.0, bond_200_209
distance bond_201_202, /4nx4//A/PHE`201/CA, /4nx4//A/CYS`202/CA
set dash_width, 1.0, bond_201_202
distance bond_201_206, /4nx4//A/PHE`201/CA, /4nx4//A/TYR`206/CA
set dash_width, 1.0, bond_201_206
distance bond_201_207, /4nx4//A/PHE`201/CA, /4nx4//A/ASN`207/CA
set dash_width, 2.0, bond_201_207
distance bond_201_208, /4nx4//A/PHE`201/CA, /4nx4//A/ALA`208/CA
set dash_width, 1.0, bond_201_208
distance bond_201_209, /4nx4//A/PHE`201/CA, /4nx4//A/PHE`209/CA
set dash_width, 1.0, bond_201_209
distance bond_202_206, /4nx4//A/CYS`202/CA, /4nx4//A/TYR`206/CA
set dash_width, 2.0, bond_202_206
distance bond_202_207, /4nx4//A/CYS`202/CA, /4nx4//A/ASN`207/CA
set dash_width, 8.0, bond_202_207
distance bond_202_208, /4nx4//A/CYS`202/CA, /4nx4//A/ALA`208/CA
set dash_width, 1.0, bond_202_208
distance bond_203_204, /4nx4//A/SER`203/CA, /4nx4//A/GLY`204/CA
set dash_width, 1.0, bond_203_204
distance bond_203_207, /4nx4//A/SER`203/CA, /4nx4//A/ASN`207/CA
set dash_width, 1.0, bond_203_207
distance bond_204_205, /4nx4//A/GLY`204/CA, /4nx4//A/CYS`205/CA
set dash_width, 1.0, bond_204_205
distance bond_205_206, /4nx4//A/CYS`205/CA, /4nx4//A/TYR`206/CA
set dash_width, 1.0, bond_205_206
distance bond_205_207, /4nx4//A/CYS`205/CA, /4nx4//A/ASN`207/CA
set dash_width, 1.0, bond_205_207
distance bond_205_370, /4nx4//A/CYS`205/CA, /4nx4//A/SER`370/CA
set dash_width, 1.0, bond_205_370
distance bond_206_207, /4nx4//A/TYR`206/CA, /4nx4//A/ASN`207/CA
set dash_width, 2.0, bond_206_207
distance bond_206_370, /4nx4//A/TYR`206/CA, /4nx4//A/SER`370/CA
set dash_width, 2.0, bond_206_370
distance bond_207_208, /4nx4//A/ASN`207/CA, /4nx4//A/ALA`208/CA
set dash_width, 1.0, bond_207_208
distance bond_207_229, /4nx4//A/ASN`207/CA, /4nx4//A/HIS`229/CA
set dash_width, 1.0, bond_207_229
distance bond_208_209, /4nx4//A/ALA`208/CA, /4nx4//A/PHE`209/CA
set dash_width, 1.0, bond_208_209
distance bond_208_229, /4nx4//A/ALA`208/CA, /4nx4//A/HIS`229/CA
set dash_width, 1.0, bond_208_229
distance bond_209_210, /4nx4//A/PHE`209/CA, /4nx4//A/TYR`210/CA
set dash_width, 1.0, bond_209_210
distance bond_209_227, /4nx4//A/PHE`209/CA, /4nx4//A/HIS`227/CA
set dash_width, 2.0, bond_209_227
distance bond_210_211, /4nx4//A/TYR`210/CA, /4nx4//A/ALA`211/CA
set dash_width, 1.0, bond_210_211
distance bond_210_225, /4nx4//A/TYR`210/CA, /4nx4//A/SER`225/CA
set dash_width, 1.0, bond_210_225
distance bond_210_227, /4nx4//A/TYR`210/CA, /4nx4//A/HIS`227/CA
set dash_width, 9.0, bond_210_227
distance bond_211_212, /4nx4//A/ALA`211/CA, /4nx4//A/LYS`212/CA
set dash_width, 1.0, bond_211_212
distance bond_211_214, /4nx4//A/ALA`211/CA, /4nx4//A/LYS`214/CA
set dash_width, 1.0, bond_211_214
distance bond_211_225, /4nx4//A/ALA`211/CA, /4nx4//A/SER`225/CA
set dash_width, 1.0, bond_211_225
distance bond_211_227, /4nx4//A/ALA`211/CA, /4nx4//A/HIS`227/CA
set dash_width, 1.0, bond_211_227
distance bond_212_213, /4nx4//A/LYS`212/CA, /4nx4//A/ASN`213/CA
set dash_width, 1.0, bond_212_213
distance bond_212_223, /4nx4//A/LYS`212/CA, /4nx4//A/LYS`223/CA
set dash_width, 2.0, bond_212_223
distance bond_212_224, /4nx4//A/LYS`212/CA, /4nx4//A/LYS`224/CA
set dash_width, 1.0, bond_212_224
distance bond_212_225, /4nx4//A/LYS`212/CA, /4nx4//A/SER`225/CA
set dash_width, 2.0, bond_212_225
distance bond_213_214, /4nx4//A/ASN`213/CA, /4nx4//A/LYS`214/CA
set dash_width, 1.0, bond_213_214
distance bond_213_222, /4nx4//A/ASN`213/CA, /4nx4//A/VAL`222/CA
set dash_width, 1.0, bond_213_222
distance bond_213_223, /4nx4//A/ASN`213/CA, /4nx4//A/LYS`223/CA
set dash_width, 9.0, bond_213_223
distance bond_213_224, /4nx4//A/ASN`213/CA, /4nx4//A/LYS`224/CA
set dash_width, 1.0, bond_213_224
distance bond_213_225, /4nx4//A/ASN`213/CA, /4nx4//A/SER`225/CA
set dash_width, 1.0, bond_213_225
distance bond_214_215, /4nx4//A/LYS`214/CA, /4nx4//A/CYS`215/CA
set dash_width, 2.0, bond_214_215
distance bond_214_223, /4nx4//A/LYS`214/CA, /4nx4//A/LYS`223/CA
set dash_width, 2.0, bond_214_223
distance bond_215_217, /4nx4//A/CYS`215/CA, /4nx4//A/GLU`217/CA
set dash_width, 1.0, bond_215_217
distance bond_215_223, /4nx4//A/CYS`215/CA, /4nx4//A/LYS`223/CA
set dash_width, 1.0, bond_215_223
distance bond_217_218, /4nx4//A/GLU`217/CA, /4nx4//A/PRO`218/CA
set dash_width, 1.0, bond_217_218
distance bond_219_220, /4nx4//A/ASN`219/CA, /4nx4//A/CYS`220/CA
set dash_width, 1.0, bond_219_220
distance bond_221_222, /4nx4//A/ARG`221/CA, /4nx4//A/VAL`222/CA
set dash_width, 1.0, bond_221_222
distance bond_222_223, /4nx4//A/VAL`222/CA, /4nx4//A/LYS`223/CA
set dash_width, 1.0, bond_222_223
distance bond_223_224, /4nx4//A/LYS`223/CA, /4nx4//A/LYS`224/CA
set dash_width, 1.0, bond_223_224
distance bond_223_225, /4nx4//A/LYS`223/CA, /4nx4//A/SER`225/CA
set dash_width, 2.0, bond_223_225
distance bond_224_225, /4nx4//A/LYS`224/CA, /4nx4//A/SER`225/CA
set dash_width, 2.0, bond_224_225
distance bond_225_226, /4nx4//A/SER`225/CA, /4nx4//A/LEU`226/CA
set dash_width, 1.0, bond_225_226
distance bond_226_227, /4nx4//A/LEU`226/CA, /4nx4//A/HIS`227/CA
set dash_width, 1.0, bond_226_227
distance bond_227_228, /4nx4//A/HIS`227/CA, /4nx4//A/GLY`228/CA
set dash_width, 1.0, bond_227_228
distance bond_228_229, /4nx4//A/GLY`228/CA, /4nx4//A/HIS`229/CA
set dash_width, 1.0, bond_228_229
distance bond_229_230, /4nx4//A/HIS`229/CA, /4nx4//A/HIS`230/CA
set dash_width, 1.0, bond_229_230
distance bond_230_231, /4nx4//A/HIS`230/CA, /4nx4//A/PRO`231/CA
set dash_width, 1.0, bond_230_231
distance bond_232_233, /4nx4//A/ARG`232/CA, /4nx4//A/ASP`233/CA
set dash_width, 1.0, bond_232_233
distance bond_232_234, /4nx4//A/ARG`232/CA, /4nx4//A/CYS`234/CA
set dash_width, 1.0, bond_232_234
distance bond_233_234, /4nx4//A/ASP`233/CA, /4nx4//A/CYS`234/CA
set dash_width, 2.0, bond_233_234
distance bond_234_235, /4nx4//A/CYS`234/CA, /4nx4//A/LEU`235/CA
set dash_width, 1.0, bond_234_235
distance bond_235_236, /4nx4//A/LEU`235/CA, /4nx4//A/PHE`236/CA
set dash_width, 1.0, bond_235_236
distance bond_236_237, /4nx4//A/PHE`236/CA, /4nx4//A/TYR`237/CA
set dash_width, 1.0, bond_236_237
distance bond_237_238, /4nx4//A/TYR`237/CA, /4nx4//A/LEU`238/CA
set dash_width, 1.0, bond_237_238
distance bond_238_239, /4nx4//A/LEU`238/CA, /4nx4//A/ARG`239/CA
set dash_width, 1.0, bond_238_239
distance bond_239_240, /4nx4//A/ARG`239/CA, /4nx4//A/ASP`240/CA
set dash_width, 1.0, bond_239_240
distance bond_239_241, /4nx4//A/ARG`239/CA, /4nx4//A/TRP`241/CA
set dash_width, 1.0, bond_239_241
distance bond_239_307, /4nx4//A/ARG`239/CA, /4nx4//A/LYS`307/CA
set dash_width, 1.0, bond_239_307
distance bond_240_241, /4nx4//A/ASP`240/CA, /4nx4//A/TRP`241/CA
set dash_width, 2.0, bond_240_241
distance bond_241_242, /4nx4//A/TRP`241/CA, /4nx4//A/THR`242/CA
set dash_width, 1.0, bond_241_242
distance bond_242_243, /4nx4//A/THR`242/CA, /4nx4//A/ALA`243/CA
set dash_width, 1.0, bond_242_243
distance bond_243_244, /4nx4//A/ALA`243/CA, /4nx4//A/LEU`244/CA
set dash_width, 1.0, bond_243_244
distance bond_244_245, /4nx4//A/LEU`244/CA, /4nx4//A/ARG`245/CA
set dash_width, 1.0, bond_244_245
distance bond_245_246, /4nx4//A/ARG`245/CA, /4nx4//A/LEU`246/CA
set dash_width, 1.0, bond_245_246
distance bond_246_247, /4nx4//A/LEU`246/CA, /4nx4//A/GLN`247/CA
set dash_width, 1.0, bond_246_247
distance bond_247_248, /4nx4//A/GLN`247/CA, /4nx4//A/LYS`248/CA
set dash_width, 1.0, bond_247_248
distance bond_248_249, /4nx4//A/LYS`248/CA, /4nx4//A/LEU`249/CA
set dash_width, 1.0, bond_248_249
distance bond_249_250, /4nx4//A/LEU`249/CA, /4nx4//A/LEU`250/CA
set dash_width, 1.0, bond_249_250
distance bond_250_251, /4nx4//A/LEU`250/CA, /4nx4//A/GLN`251/CA
set dash_width, 1.0, bond_250_251
distance bond_250_255, /4nx4//A/LEU`250/CA, /4nx4//A/VAL`255/CA
set dash_width, 1.0, bond_250_255
distance bond_251_252, /4nx4//A/GLN`251/CA, /4nx4//A/ASP`252/CA
set dash_width, 1.0, bond_251_252
distance bond_251_253, /4nx4//A/GLN`251/CA, /4nx4//A/ASN`253/CA
set dash_width, 1.0, bond_251_253
distance bond_251_255, /4nx4//A/GLN`251/CA, /4nx4//A/VAL`255/CA
set dash_width, 2.0, bond_251_255
distance bond_252_253, /4nx4//A/ASP`252/CA, /4nx4//A/ASN`253/CA
set dash_width, 1.0, bond_252_253
distance bond_253_254, /4nx4//A/ASN`253/CA, /4nx4//A/ASN`254/CA
set dash_width, 1.0, bond_253_254
distance bond_254_255, /4nx4//A/ASN`254/CA, /4nx4//A/VAL`255/CA
set dash_width, 2.0, bond_254_255
distance bond_255_256, /4nx4//A/VAL`255/CA, /4nx4//A/MET`256/CA
set dash_width, 1.0, bond_255_256
distance bond_256_257, /4nx4//A/MET`256/CA, /4nx4//A/PHE`257/CA
set dash_width, 1.0, bond_256_257
distance bond_257_258, /4nx4//A/PHE`257/CA, /4nx4//A/ASN`258/CA
set dash_width, 1.0, bond_257_258
distance bond_258_260, /4nx4//A/ASN`258/CA, /4nx4//A/GLU`260/CA
set dash_width, 1.0, bond_258_260
distance bond_259_260, /4nx4//A/THR`259/CA, /4nx4//A/GLU`260/CA
set dash_width, 3.0, bond_259_260
distance bond_259_297, /4nx4//A/THR`259/CA, /4nx4//A/GLY`297/CA
set dash_width, 2.0, bond_259_297
distance bond_259_298, /4nx4//A/THR`259/CA, /4nx4//A/TYR`298/CA
set dash_width, 1.0, bond_259_298
distance bond_260_261, /4nx4//A/GLU`260/CA, /4nx4//A/PRO`261/CA
set dash_width, 1.0, bond_260_261
distance bond_260_297, /4nx4//A/GLU`260/CA, /4nx4//A/GLY`297/CA
set dash_width, 1.0, bond_260_297
distance bond_260_298, /4nx4//A/GLU`260/CA, /4nx4//A/TYR`298/CA
set dash_width, 1.0, bond_260_298
distance bond_260_299, /4nx4//A/GLU`260/CA, /4nx4//A/ALA`299/CA
set dash_width, 1.0, bond_260_299
distance bond_264_265, /4nx4//A/GLY`264/CA, /4nx4//A/ALA`265/CA
set dash_width, 2.0, bond_264_265
distance bond_265_266, /4nx4//A/ALA`265/CA, /4nx4//A/ARG`266/CA
set dash_width, 1.0, bond_265_266
distance bond_266_267, /4nx4//A/ARG`266/CA, /4nx4//A/ALA`267/CA
set dash_width, 1.0, bond_266_267
distance bond_267_268, /4nx4//A/ALA`267/CA, /4nx4//A/VAL`268/CA
set dash_width, 1.0, bond_267_268
distance bond_268_269, /4nx4//A/VAL`268/CA, /4nx4//A/PRO`269/CA
set dash_width, 1.0, bond_268_269
distance bond_268_271, /4nx4//A/VAL`268/CA, /4nx4//A/GLY`271/CA
set dash_width, 1.0, bond_268_271
distance bond_268_272, /4nx4//A/VAL`268/CA, /4nx4//A/GLY`272/CA
set dash_width, 1.0, bond_268_272
distance bond_270_271, /4nx4//A/GLY`270/CA, /4nx4//A/GLY`271/CA
set dash_width, 2.0, bond_270_271
distance bond_271_272, /4nx4//A/GLY`271/CA, /4nx4//A/GLY`272/CA
set dash_width, 1.0, bond_271_272
distance bond_272_273, /4nx4//A/GLY`272/CA, /4nx4//A/CYS`273/CA
set dash_width, 1.0, bond_272_273
distance bond_272_291, /4nx4//A/GLY`272/CA, /4nx4//A/GLY`291/CA
set dash_width, 1.0, bond_272_291
distance bond_272_292, /4nx4//A/GLY`272/CA, /4nx4//A/LYS`292/CA
set dash_width, 2.0, bond_272_292
distance bond_272_300, /4nx4//A/GLY`272/CA, /4nx4//A/GLY`300/CA
set dash_width, 1.0, bond_272_300
distance bond_273_290, /4nx4//A/CYS`273/CA, /4nx4//A/CYS`290/CA
set dash_width, 1.0, bond_273_290
distance bond_273_291, /4nx4//A/CYS`273/CA, /4nx4//A/GLY`291/CA
set dash_width, 2.0, bond_273_291
distance bond_273_292, /4nx4//A/CYS`273/CA, /4nx4//A/LYS`292/CA
set dash_width, 8.0, bond_273_292
distance bond_273_293, /4nx4//A/CYS`273/CA, /4nx4//A/GLU`293/CA
set dash_width, 1.0, bond_273_293
distance bond_273_300, /4nx4//A/CYS`273/CA, /4nx4//A/GLY`300/CA
set dash_width, 2.0, bond_273_300
distance bond_273_301, /4nx4//A/CYS`273/CA, /4nx4//A/LEU`301/CA
set dash_width, 1.0, bond_273_301
distance bond_274_275, /4nx4//A/ARG`274/CA, /4nx4//A/VAL`275/CA
set dash_width, 2.0, bond_274_275
distance bond_274_292, /4nx4//A/ARG`274/CA, /4nx4//A/LYS`292/CA
set dash_width, 1.0, bond_274_292
distance bond_274_300, /4nx4//A/ARG`274/CA, /4nx4//A/GLY`300/CA
set dash_width, 2.0, bond_274_300
distance bond_275_276, /4nx4//A/VAL`275/CA, /4nx4//A/ILE`276/CA
set dash_width, 1.0, bond_275_276
distance bond_275_288, /4nx4//A/VAL`275/CA, /4nx4//A/GLU`288/CA
set dash_width, 1.0, bond_275_288
distance bond_275_300, /4nx4//A/VAL`275/CA, /4nx4//A/GLY`300/CA
set dash_width, 1.0, bond_275_300
distance bond_276_277, /4nx4//A/ILE`276/CA, /4nx4//A/GLU`277/CA
set dash_width, 1.0, bond_276_277
distance bond_276_288, /4nx4//A/ILE`276/CA, /4nx4//A/GLU`288/CA
set dash_width, 2.0, bond_276_288
distance bond_276_289, /4nx4//A/ILE`276/CA, /4nx4//A/ALA`289/CA
set dash_width, 1.0, bond_276_289
distance bond_277_278, /4nx4//A/GLU`277/CA, /4nx4//A/GLN`278/CA
set dash_width, 1.0, bond_277_278
distance bond_277_286, /4nx4//A/GLU`277/CA, /4nx4//A/ARG`286/CA
set dash_width, 1.0, bond_277_286
distance bond_277_287, /4nx4//A/GLU`277/CA, /4nx4//A/ASP`287/CA
set dash_width, 1.0, bond_277_287
distance bond_277_288, /4nx4//A/GLU`277/CA, /4nx4//A/GLU`288/CA
set dash_width, 10.0, bond_277_288
distance bond_277_289, /4nx4//A/GLU`277/CA, /4nx4//A/ALA`289/CA
set dash_width, 1.0, bond_277_289
distance bond_278_279, /4nx4//A/GLN`278/CA, /4nx4//A/LYS`279/CA
set dash_width, 1.0, bond_278_279
distance bond_278_286, /4nx4//A/GLN`278/CA, /4nx4//A/ARG`286/CA
set dash_width, 2.0, bond_278_286
distance bond_278_288, /4nx4//A/GLN`278/CA, /4nx4//A/GLU`288/CA
set dash_width, 2.0, bond_278_288
distance bond_279_280, /4nx4//A/LYS`279/CA, /4nx4//A/GLU`280/CA
set dash_width, 1.0, bond_279_280
distance bond_279_285, /4nx4//A/LYS`279/CA, /4nx4//A/LEU`285/CA
set dash_width, 1.0, bond_279_285
distance bond_279_286, /4nx4//A/LYS`279/CA, /4nx4//A/ARG`286/CA
set dash_width, 7.0, bond_279_286
distance bond_279_288, /4nx4//A/LYS`279/CA, /4nx4//A/GLU`288/CA
set dash_width, 1.0, bond_279_288
distance bond_280_281, /4nx4//A/GLU`280/CA, /4nx4//A/VAL`281/CA
set dash_width, 1.0, bond_280_281
distance bond_280_284, /4nx4//A/GLU`280/CA, /4nx4//A/GLY`284/CA
set dash_width, 2.0, bond_280_284
distance bond_280_286, /4nx4//A/GLU`280/CA, /4nx4//A/ARG`286/CA
set dash_width, 1.0, bond_280_286
distance bond_281_282, /4nx4//A/VAL`281/CA, /4nx4//A/PRO`282/CA
set dash_width, 1.0, bond_281_282
distance bond_281_284, /4nx4//A/VAL`281/CA, /4nx4//A/GLY`284/CA
set dash_width, 8.0, bond_281_284
distance bond_281_285, /4nx4//A/VAL`281/CA, /4nx4//A/LEU`285/CA
set dash_width, 1.0, bond_281_285
distance bond_283_284, /4nx4//A/ASN`283/CA, /4nx4//A/GLY`284/CA
set dash_width, 2.0, bond_283_284
distance bond_284_285, /4nx4//A/GLY`284/CA, /4nx4//A/LEU`285/CA
set dash_width, 1.0, bond_284_285
distance bond_285_286, /4nx4//A/LEU`285/CA, /4nx4//A/ARG`286/CA
set dash_width, 1.0, bond_285_286
distance bond_286_287, /4nx4//A/ARG`286/CA, /4nx4//A/ASP`287/CA
set dash_width, 1.0, bond_286_287
distance bond_287_288, /4nx4//A/ASP`287/CA, /4nx4//A/GLU`288/CA
set dash_width, 1.0, bond_287_288
distance bond_288_289, /4nx4//A/GLU`288/CA, /4nx4//A/ALA`289/CA
set dash_width, 1.0, bond_288_289
distance bond_289_290, /4nx4//A/ALA`289/CA, /4nx4//A/CYS`290/CA
set dash_width, 1.0, bond_289_290
distance bond_290_291, /4nx4//A/CYS`290/CA, /4nx4//A/GLY`291/CA
set dash_width, 1.0, bond_290_291
distance bond_291_292, /4nx4//A/GLY`291/CA, /4nx4//A/LYS`292/CA
set dash_width, 2.0, bond_291_292
distance bond_292_293, /4nx4//A/LYS`292/CA, /4nx4//A/GLU`293/CA
set dash_width, 1.0, bond_292_293
distance bond_293_294, /4nx4//A/GLU`293/CA, /4nx4//A/THR`294/CA
set dash_width, 1.0, bond_293_294
distance bond_294_295, /4nx4//A/THR`294/CA, /4nx4//A/PRO`295/CA
set dash_width, 1.0, bond_294_295
distance bond_294_300, /4nx4//A/THR`294/CA, /4nx4//A/GLY`300/CA
set dash_width, 1.0, bond_294_300
distance bond_294_301, /4nx4//A/THR`294/CA, /4nx4//A/LEU`301/CA
set dash_width, 2.0, bond_294_301
distance bond_296_297, /4nx4//A/ALA`296/CA, /4nx4//A/GLY`297/CA
set dash_width, 1.0, bond_296_297
distance bond_297_298, /4nx4//A/GLY`297/CA, /4nx4//A/TYR`298/CA
set dash_width, 1.0, bond_297_298
distance bond_298_299, /4nx4//A/TYR`298/CA, /4nx4//A/ALA`299/CA
set dash_width, 1.0, bond_298_299
distance bond_298_301, /4nx4//A/TYR`298/CA, /4nx4//A/LEU`301/CA
set dash_width, 1.0, bond_298_301
distance bond_299_300, /4nx4//A/ALA`299/CA, /4nx4//A/GLY`300/CA
set dash_width, 1.0, bond_299_300
distance bond_299_301, /4nx4//A/ALA`299/CA, /4nx4//A/LEU`301/CA
set dash_width, 1.0, bond_299_301
distance bond_300_301, /4nx4//A/GLY`300/CA, /4nx4//A/LEU`301/CA
set dash_width, 2.0, bond_300_301
distance bond_301_302, /4nx4//A/LEU`301/CA, /4nx4//A/CYS`302/CA
set dash_width, 2.0, bond_301_302
distance bond_302_303, /4nx4//A/CYS`302/CA, /4nx4//A/GLN`303/CA
set dash_width, 1.0, bond_302_303
distance bond_302_305, /4nx4//A/CYS`302/CA, /4nx4//A/HIS`305/CA
set dash_width, 1.0, bond_302_305
distance bond_303_304, /4nx4//A/GLN`303/CA, /4nx4//A/ALA`304/CA
set dash_width, 1.0, bond_303_304
distance bond_304_305, /4nx4//A/ALA`304/CA, /4nx4//A/HIS`305/CA
set dash_width, 1.0, bond_304_305
distance bond_305_306, /4nx4//A/HIS`305/CA, /4nx4//A/TYR`306/CA
set dash_width, 1.0, bond_305_306
distance bond_306_307, /4nx4//A/TYR`306/CA, /4nx4//A/LYS`307/CA
set dash_width, 1.0, bond_306_307
distance bond_307_308, /4nx4//A/LYS`307/CA, /4nx4//A/GLU`308/CA
set dash_width, 1.0, bond_307_308
distance bond_308_309, /4nx4//A/GLU`308/CA, /4nx4//A/TYR`309/CA
set dash_width, 1.0, bond_308_309
distance bond_309_310, /4nx4//A/TYR`309/CA, /4nx4//A/LEU`310/CA
set dash_width, 1.0, bond_309_310
distance bond_310_311, /4nx4//A/LEU`310/CA, /4nx4//A/VAL`311/CA
set dash_width, 1.0, bond_310_311
distance bond_311_312, /4nx4//A/VAL`311/CA, /4nx4//A/SER`312/CA
set dash_width, 1.0, bond_311_312
distance bond_312_313, /4nx4//A/SER`312/CA, /4nx4//A/LEU`313/CA
set dash_width, 1.0, bond_312_313
distance bond_313_314, /4nx4//A/LEU`313/CA, /4nx4//A/ILE`314/CA
set dash_width, 1.0, bond_313_314
distance bond_314_315, /4nx4//A/ILE`314/CA, /4nx4//A/ASN`315/CA
set dash_width, 1.0, bond_314_315
distance bond_315_316, /4nx4//A/ASN`315/CA, /4nx4//A/ALA`316/CA
set dash_width, 1.0, bond_315_316
distance bond_315_317, /4nx4//A/ASN`315/CA, /4nx4//A/HIS`317/CA
set dash_width, 1.0, bond_315_317
distance bond_315_319, /4nx4//A/ASN`315/CA, /4nx4//A/LEU`319/CA
set dash_width, 1.0, bond_315_319
distance bond_316_317, /4nx4//A/ALA`316/CA, /4nx4//A/HIS`317/CA
set dash_width, 1.0, bond_316_317
distance bond_316_372, /4nx4//A/ALA`316/CA, /4nx4//A/PRO`372/CA
set dash_width, 2.0, bond_316_372
distance bond_316_374, /4nx4//A/ALA`316/CA, /4nx4//A/ARG`374/CA
set dash_width, 1.0, bond_316_374
distance bond_317_318, /4nx4//A/HIS`317/CA, /4nx4//A/SER`318/CA
set dash_width, 1.0, bond_317_318
distance bond_317_374, /4nx4//A/HIS`317/CA, /4nx4//A/ARG`374/CA
set dash_width, 1.0, bond_317_374
distance bond_318_319, /4nx4//A/SER`318/CA, /4nx4//A/LEU`319/CA
set dash_width, 1.0, bond_318_319
distance bond_318_374, /4nx4//A/SER`318/CA, /4nx4//A/ARG`374/CA
set dash_width, 1.0, bond_318_374
distance bond_319_320, /4nx4//A/LEU`319/CA, /4nx4//A/ASP`320/CA
set dash_width, 1.0, bond_319_320
distance bond_320_321, /4nx4//A/ASP`320/CA, /4nx4//A/PRO`321/CA
set dash_width, 1.0, bond_320_321
distance bond_322_323, /4nx4//A/ALA`322/CA, /4nx4//A/THR`323/CA
set dash_width, 1.0, bond_322_323
distance bond_323_324, /4nx4//A/THR`323/CA, /4nx4//A/LEU`324/CA
set dash_width, 1.0, bond_323_324
distance bond_323_325, /4nx4//A/THR`323/CA, /4nx4//A/TYR`325/CA
set dash_width, 1.0, bond_323_325
distance bond_324_325, /4nx4//A/LEU`324/CA, /4nx4//A/TYR`325/CA
set dash_width, 2.0, bond_324_325
distance bond_325_326, /4nx4//A/TYR`325/CA, /4nx4//A/GLU`326/CA
set dash_width, 1.0, bond_325_326
distance bond_326_327, /4nx4//A/GLU`326/CA, /4nx4//A/VAL`327/CA
set dash_width, 1.0, bond_326_327
distance bond_326_329, /4nx4//A/GLU`326/CA, /4nx4//A/GLU`329/CA
set dash_width, 1.0, bond_326_329
distance bond_327_328, /4nx4//A/VAL`327/CA, /4nx4//A/GLU`328/CA
set dash_width, 1.0, bond_327_328
distance bond_328_329, /4nx4//A/GLU`328/CA, /4nx4//A/GLU`329/CA
set dash_width, 1.0, bond_328_329
distance bond_329_330, /4nx4//A/GLU`329/CA, /4nx4//A/LEU`330/CA
set dash_width, 1.0, bond_329_330
distance bond_330_331, /4nx4//A/LEU`330/CA, /4nx4//A/GLU`331/CA
set dash_width, 1.0, bond_330_331
distance bond_331_332, /4nx4//A/GLU`331/CA, /4nx4//A/THR`332/CA
set dash_width, 1.0, bond_331_332
distance bond_332_333, /4nx4//A/THR`332/CA, /4nx4//A/ALA`333/CA
set dash_width, 1.0, bond_332_333
distance bond_333_334, /4nx4//A/ALA`333/CA, /4nx4//A/THR`334/CA
set dash_width, 1.0, bond_333_334
distance bond_334_335, /4nx4//A/THR`334/CA, /4nx4//A/GLU`335/CA
set dash_width, 1.0, bond_334_335
distance bond_334_340, /4nx4//A/THR`334/CA, /4nx4//A/VAL`340/CA
set dash_width, 1.0, bond_334_340
distance bond_335_336, /4nx4//A/GLU`335/CA, /4nx4//A/ARG`336/CA
set dash_width, 1.0, bond_335_336
distance bond_335_338, /4nx4//A/GLU`335/CA, /4nx4//A/LEU`338/CA
set dash_width, 1.0, bond_335_338
distance bond_335_340, /4nx4//A/GLU`335/CA, /4nx4//A/VAL`340/CA
set dash_width, 4.0, bond_335_340
distance bond_336_337, /4nx4//A/ARG`336/CA, /4nx4//A/TYR`337/CA
set dash_width, 1.0, bond_336_337
distance bond_336_340, /4nx4//A/ARG`336/CA, /4nx4//A/VAL`340/CA
set dash_width, 1.0, bond_336_340
distance bond_337_338, /4nx4//A/TYR`337/CA, /4nx4//A/LEU`338/CA
set dash_width, 1.0, bond_337_338
distance bond_338_339, /4nx4//A/LEU`338/CA, /4nx4//A/HIS`339/CA
set dash_width, 1.0, bond_338_339
distance bond_338_340, /4nx4//A/LEU`338/CA, /4nx4//A/VAL`340/CA
set dash_width, 1.0, bond_338_340
distance bond_339_340, /4nx4//A/HIS`339/CA, /4nx4//A/VAL`340/CA
set dash_width, 2.0, bond_339_340
distance bond_340_341, /4nx4//A/VAL`340/CA, /4nx4//A/ARG`341/CA
set dash_width, 1.0, bond_340_341
distance bond_341_342, /4nx4//A/ARG`341/CA, /4nx4//A/PRO`342/CA
set dash_width, 1.0, bond_341_342
distance bond_343_344, /4nx4//A/GLN`343/CA, /4nx4//A/PRO`344/CA
set dash_width, 1.0, bond_343_344
distance bond_345_346, /4nx4//A/LEU`345/CA, /4nx4//A/ALA`346/CA
set dash_width, 1.0, bond_345_346
distance bond_345_348, /4nx4//A/LEU`345/CA, /4nx4//A/GLU`348/CA
set dash_width, 1.0, bond_345_348
distance bond_346_347, /4nx4//A/ALA`346/CA, /4nx4//A/GLY`347/CA
set dash_width, 1.0, bond_346_347
distance bond_347_348, /4nx4//A/GLY`347/CA, /4nx4//A/GLU`348/CA
set dash_width, 2.0, bond_347_348
distance bond_348_349, /4nx4//A/GLU`348/CA, /4nx4//A/ASP`349/CA
set dash_width, 1.0, bond_348_349
distance bond_349_350, /4nx4//A/ASP`349/CA, /4nx4//A/PRO`350/CA
set dash_width, 1.0, bond_349_350
distance bond_352_353, /4nx4//A/ALA`352/CA, /4nx4//A/TYR`353/CA
set dash_width, 1.0, bond_352_353
distance bond_353_354, /4nx4//A/TYR`353/CA, /4nx4//A/GLN`354/CA
set dash_width, 1.0, bond_353_354
distance bond_354_355, /4nx4//A/GLN`354/CA, /4nx4//A/ALA`355/CA
set dash_width, 1.0, bond_354_355
distance bond_355_356, /4nx4//A/ALA`355/CA, /4nx4//A/ARG`356/CA
set dash_width, 1.0, bond_355_356
distance bond_356_357, /4nx4//A/ARG`356/CA, /4nx4//A/LEU`357/CA
set dash_width, 1.0, bond_356_357
distance bond_357_358, /4nx4//A/LEU`357/CA, /4nx4//A/LEU`358/CA
set dash_width, 1.0, bond_357_358
distance bond_358_359, /4nx4//A/LEU`358/CA, /4nx4//A/GLN`359/CA
set dash_width, 1.0, bond_358_359
distance bond_359_360, /4nx4//A/GLN`359/CA, /4nx4//A/LYS`360/CA
set dash_width, 1.0, bond_359_360
distance bond_360_361, /4nx4//A/LYS`360/CA, /4nx4//A/LEU`361/CA
set dash_width, 1.0, bond_360_361
distance bond_361_362, /4nx4//A/LEU`361/CA, /4nx4//A/THR`362/CA
set dash_width, 1.0, bond_361_362
distance bond_362_363, /4nx4//A/THR`362/CA, /4nx4//A/GLU`363/CA
set dash_width, 1.0, bond_362_363
distance bond_362_365, /4nx4//A/THR`362/CA, /4nx4//A/VAL`365/CA
set dash_width, 2.0, bond_362_365
distance bond_363_364, /4nx4//A/GLU`363/CA, /4nx4//A/GLU`364/CA
set dash_width, 1.0, bond_363_364
distance bond_363_365, /4nx4//A/GLU`363/CA, /4nx4//A/VAL`365/CA
set dash_width, 1.0, bond_363_365
distance bond_364_365, /4nx4//A/GLU`364/CA, /4nx4//A/VAL`365/CA
set dash_width, 2.0, bond_364_365
distance bond_365_366, /4nx4//A/VAL`365/CA, /4nx4//A/PRO`366/CA
set dash_width, 1.0, bond_365_366
distance bond_367_368, /4nx4//A/LEU`367/CA, /4nx4//A/GLY`368/CA
set dash_width, 1.0, bond_367_368
distance bond_369_370, /4nx4//A/GLN`369/CA, /4nx4//A/SER`370/CA
set dash_width, 1.0, bond_369_370
distance bond_369_371, /4nx4//A/GLN`369/CA, /4nx4//A/ILE`371/CA
set dash_width, 1.0, bond_369_371
distance bond_370_371, /4nx4//A/SER`370/CA, /4nx4//A/ILE`371/CA
set dash_width, 2.0, bond_370_371
distance bond_371_372, /4nx4//A/ILE`371/CA, /4nx4//A/PRO`372/CA
set dash_width, 1.0, bond_371_372
distance bond_373_374, /4nx4//A/ARG`373/CA, /4nx4//A/ARG`374/CA
set dash_width, 1.0, bond_373_374
distance bond_374_375, /4nx4//A/ARG`374/CA, /4nx4//A/ARG`375/CA
set dash_width, 1.0, bond_374_375
distance bond_375_376, /4nx4//A/ARG`375/CA, /4nx4//A/LYS`376/CA
set dash_width, 2.0, bond_375_376

# Set B-factors based on heavy atom contacts
alter 4nx4 and resid 1 and name CA, b=21.0
alter 4nx4 and resid 2 and name CA, b=20.0
alter 4nx4 and resid 3 and name CA, b=34.0
alter 4nx4 and resid 4 and name CA, b=22.0
alter 4nx4 and resid 5 and name CA, b=16.0
alter 4nx4 and resid 6 and name CA, b=9.0
alter 4nx4 and resid 7 and name CA, b=17.0
alter 4nx4 and resid 8 and name CA, b=17.0
alter 4nx4 and resid 9 and name CA, b=14.0
alter 4nx4 and resid 10 and name CA, b=14.0
alter 4nx4 and resid 11 and name CA, b=14.0
alter 4nx4 and resid 12 and name CA, b=2.0
alter 4nx4 and resid 13 and name CA, b=10.0
alter 4nx4 and resid 14 and name CA, b=26.0
alter 4nx4 and resid 15 and name CA, b=27.0
alter 4nx4 and resid 16 and name CA, b=18.0
alter 4nx4 and resid 17 and name CA, b=24.0
alter 4nx4 and resid 18 and name CA, b=7.0
alter 4nx4 and resid 19 and name CA, b=9.0
alter 4nx4 and resid 20 and name CA, b=16.0
alter 4nx4 and resid 21 and name CA, b=7.0
alter 4nx4 and resid 22 and name CA, b=7.0
alter 4nx4 and resid 23 and name CA, b=20.0
alter 4nx4 and resid 24 and name CA, b=29.0
alter 4nx4 and resid 25 and name CA, b=36.0
alter 4nx4 and resid 26 and name CA, b=15.0
alter 4nx4 and resid 27 and name CA, b=20.0
alter 4nx4 and resid 28 and name CA, b=31.0
alter 4nx4 and resid 29 and name CA, b=29.0
alter 4nx4 and resid 30 and name CA, b=21.0
alter 4nx4 and resid 31 and name CA, b=21.0
alter 4nx4 and resid 32 and name CA, b=24.0
alter 4nx4 and resid 33 and name CA, b=25.0
alter 4nx4 and resid 34 and name CA, b=23.0
alter 4nx4 and resid 35 and name CA, b=35.0
alter 4nx4 and resid 36 and name CA, b=27.0
alter 4nx4 and resid 37 and name CA, b=20.0
alter 4nx4 and resid 38 and name CA, b=15.0
alter 4nx4 and resid 39 and name CA, b=17.0
alter 4nx4 and resid 40 and name CA, b=13.0
alter 4nx4 and resid 41 and name CA, b=13.0
alter 4nx4 and resid 42 and name CA, b=9.0
alter 4nx4 and resid 43 and name CA, b=16.0
alter 4nx4 and resid 44 and name CA, b=24.0
alter 4nx4 and resid 45 and name CA, b=31.0
alter 4nx4 and resid 46 and name CA, b=34.0
alter 4nx4 and resid 47 and name CA, b=16.0
alter 4nx4 and resid 48 and name CA, b=9.0
alter 4nx4 and resid 49 and name CA, b=17.0
alter 4nx4 and resid 50 and name CA, b=16.0
alter 4nx4 and resid 51 and name CA, b=6.0
alter 4nx4 and resid 52 and name CA, b=6.0
alter 4nx4 and resid 53 and name CA, b=7.0
alter 4nx4 and resid 54 and name CA, b=19.0
alter 4nx4 and resid 55 and name CA, b=8.0
alter 4nx4 and resid 56 and name CA, b=5.0
alter 4nx4 and resid 57 and name CA, b=13.0
alter 4nx4 and resid 58 and name CA, b=28.0
alter 4nx4 and resid 59 and name CA, b=17.0
alter 4nx4 and resid 60 and name CA, b=15.0
alter 4nx4 and resid 61 and name CA, b=18.0
alter 4nx4 and resid 62 and name CA, b=24.0
alter 4nx4 and resid 63 and name CA, b=19.0
alter 4nx4 and resid 64 and name CA, b=19.0
alter 4nx4 and resid 65 and name CA, b=31.0
alter 4nx4 and resid 66 and name CA, b=29.0
alter 4nx4 and resid 67 and name CA, b=24.0
alter 4nx4 and resid 68 and name CA, b=21.0
alter 4nx4 and resid 69 and name CA, b=30.0
alter 4nx4 and resid 70 and name CA, b=31.0
alter 4nx4 and resid 71 and name CA, b=14.0
alter 4nx4 and resid 72 and name CA, b=13.0
alter 4nx4 and resid 73 and name CA, b=29.0
alter 4nx4 and resid 74 and name CA, b=16.0
alter 4nx4 and resid 75 and name CA, b=7.0
alter 4nx4 and resid 76 and name CA, b=20.0
alter 4nx4 and resid 77 and name CA, b=27.0
alter 4nx4 and resid 78 and name CA, b=16.0
alter 4nx4 and resid 79 and name CA, b=21.0
alter 4nx4 and resid 80 and name CA, b=28.0
alter 4nx4 and resid 81 and name CA, b=27.0
alter 4nx4 and resid 82 and name CA, b=21.0
alter 4nx4 and resid 83 and name CA, b=24.0
alter 4nx4 and resid 84 and name CA, b=26.0
alter 4nx4 and resid 85 and name CA, b=23.0
alter 4nx4 and resid 86 and name CA, b=26.0
alter 4nx4 and resid 87 and name CA, b=20.0
alter 4nx4 and resid 88 and name CA, b=20.0
alter 4nx4 and resid 89 and name CA, b=21.0
alter 4nx4 and resid 90 and name CA, b=21.0
alter 4nx4 and resid 91 and name CA, b=15.0
alter 4nx4 and resid 92 and name CA, b=16.0
alter 4nx4 and resid 93 and name CA, b=11.0
alter 4nx4 and resid 94 and name CA, b=28.0
alter 4nx4 and resid 95 and name CA, b=29.0
alter 4nx4 and resid 96 and name CA, b=18.0
alter 4nx4 and resid 97 and name CA, b=32.0
alter 4nx4 and resid 98 and name CA, b=19.0
alter 4nx4 and resid 99 and name CA, b=12.0
alter 4nx4 and resid 100 and name CA, b=10.0
alter 4nx4 and resid 101 and name CA, b=16.0
alter 4nx4 and resid 102 and name CA, b=26.0
alter 4nx4 and resid 103 and name CA, b=28.0
alter 4nx4 and resid 104 and name CA, b=27.0
alter 4nx4 and resid 105 and name CA, b=17.0
alter 4nx4 and resid 106 and name CA, b=20.0
alter 4nx4 and resid 107 and name CA, b=9.0
alter 4nx4 and resid 108 and name CA, b=2.0
alter 4nx4 and resid 109 and name CA, b=8.0
alter 4nx4 and resid 110 and name CA, b=6.0
alter 4nx4 and resid 111 and name CA, b=15.0
alter 4nx4 and resid 112 and name CA, b=28.0
alter 4nx4 and resid 113 and name CA, b=43.0
alter 4nx4 and resid 114 and name CA, b=25.0
alter 4nx4 and resid 115 and name CA, b=33.0
alter 4nx4 and resid 116 and name CA, b=13.0
alter 4nx4 and resid 117 and name CA, b=12.0
alter 4nx4 and resid 118 and name CA, b=18.0
alter 4nx4 and resid 119 and name CA, b=19.0
alter 4nx4 and resid 120 and name CA, b=20.0
alter 4nx4 and resid 121 and name CA, b=36.0
alter 4nx4 and resid 122 and name CA, b=46.0
alter 4nx4 and resid 123 and name CA, b=30.0
alter 4nx4 and resid 124 and name CA, b=13.0
alter 4nx4 and resid 125 and name CA, b=11.0
alter 4nx4 and resid 126 and name CA, b=18.0
alter 4nx4 and resid 127 and name CA, b=20.0
alter 4nx4 and resid 128 and name CA, b=15.0
alter 4nx4 and resid 129 and name CA, b=17.0
alter 4nx4 and resid 130 and name CA, b=18.0
alter 4nx4 and resid 131 and name CA, b=4.0
alter 4nx4 and resid 132 and name CA, b=9.0
alter 4nx4 and resid 133 and name CA, b=32.0
alter 4nx4 and resid 134 and name CA, b=19.0
alter 4nx4 and resid 135 and name CA, b=6.0
alter 4nx4 and resid 136 and name CA, b=13.0
alter 4nx4 and resid 137 and name CA, b=18.0
alter 4nx4 and resid 138 and name CA, b=21.0
alter 4nx4 and resid 139 and name CA, b=11.0
alter 4nx4 and resid 140 and name CA, b=18.0
alter 4nx4 and resid 141 and name CA, b=31.0
alter 4nx4 and resid 142 and name CA, b=21.0
alter 4nx4 and resid 143 and name CA, b=22.0
alter 4nx4 and resid 144 and name CA, b=21.0
alter 4nx4 and resid 145 and name CA, b=26.0
alter 4nx4 and resid 146 and name CA, b=18.0
alter 4nx4 and resid 147 and name CA, b=22.0
alter 4nx4 and resid 148 and name CA, b=30.0
alter 4nx4 and resid 149 and name CA, b=26.0
alter 4nx4 and resid 150 and name CA, b=7.0
alter 4nx4 and resid 151 and name CA, b=18.0
alter 4nx4 and resid 152 and name CA, b=25.0
alter 4nx4 and resid 153 and name CA, b=19.0
alter 4nx4 and resid 154 and name CA, b=18.0
alter 4nx4 and resid 155 and name CA, b=26.0
alter 4nx4 and resid 156 and name CA, b=18.0
alter 4nx4 and resid 157 and name CA, b=21.0
alter 4nx4 and resid 158 and name CA, b=20.0
alter 4nx4 and resid 159 and name CA, b=22.0
alter 4nx4 and resid 160 and name CA, b=25.0
alter 4nx4 and resid 161 and name CA, b=16.0
alter 4nx4 and resid 162 and name CA, b=13.0
alter 4nx4 and resid 163 and name CA, b=22.0
alter 4nx4 and resid 164 and name CA, b=30.0
alter 4nx4 and resid 165 and name CA, b=29.0
alter 4nx4 and resid 166 and name CA, b=27.0
alter 4nx4 and resid 167 and name CA, b=41.0
alter 4nx4 and resid 168 and name CA, b=11.0
alter 4nx4 and resid 169 and name CA, b=12.0
alter 4nx4 and resid 170 and name CA, b=17.0
alter 4nx4 and resid 171 and name CA, b=17.0
alter 4nx4 and resid 172 and name CA, b=14.0
alter 4nx4 and resid 173 and name CA, b=23.0
alter 4nx4 and resid 174 and name CA, b=17.0
alter 4nx4 and resid 175 and name CA, b=18.0
alter 4nx4 and resid 176 and name CA, b=15.0
alter 4nx4 and resid 177 and name CA, b=19.0
alter 4nx4 and resid 178 and name CA, b=18.0
alter 4nx4 and resid 179 and name CA, b=11.0
alter 4nx4 and resid 180 and name CA, b=2.0
alter 4nx4 and resid 181 and name CA, b=11.0
alter 4nx4 and resid 182 and name CA, b=19.0
alter 4nx4 and resid 183 and name CA, b=37.0
alter 4nx4 and resid 184 and name CA, b=18.0
alter 4nx4 and resid 185 and name CA, b=32.0
alter 4nx4 and resid 186 and name CA, b=18.0
alter 4nx4 and resid 187 and name CA, b=12.0
alter 4nx4 and resid 188 and name CA, b=10.0
alter 4nx4 and resid 189 and name CA, b=17.0
alter 4nx4 and resid 190 and name CA, b=20.0
alter 4nx4 and resid 191 and name CA, b=25.0
alter 4nx4 and resid 192 and name CA, b=43.0
alter 4nx4 and resid 193 and name CA, b=44.0
alter 4nx4 and resid 194 and name CA, b=36.0
alter 4nx4 and resid 195 and name CA, b=17.0
alter 4nx4 and resid 196 and name CA, b=26.0
alter 4nx4 and resid 197 and name CA, b=28.0
alter 4nx4 and resid 198 and name CA, b=21.0
alter 4nx4 and resid 199 and name CA, b=19.0
alter 4nx4 and resid 200 and name CA, b=27.0
alter 4nx4 and resid 201 and name CA, b=25.0
alter 4nx4 and resid 202 and name CA, b=26.0
alter 4nx4 and resid 203 and name CA, b=19.0
alter 4nx4 and resid 204 and name CA, b=22.0
alter 4nx4 and resid 205 and name CA, b=15.0
alter 4nx4 and resid 206 and name CA, b=28.0
alter 4nx4 and resid 207 and name CA, b=11.0
alter 4nx4 and resid 208 and name CA, b=7.0
alter 4nx4 and resid 209 and name CA, b=20.0
alter 4nx4 and resid 210 and name CA, b=3.0
alter 4nx4 and resid 211 and name CA, b=10.0
alter 4nx4 and resid 212 and name CA, b=17.0
alter 4nx4 and resid 213 and name CA, b=21.0
alter 4nx4 and resid 214 and name CA, b=21.0
alter 4nx4 and resid 215 and name CA, b=16.0
alter 4nx4 and resid 216 and name CA, b=25.0
alter 4nx4 and resid 217 and name CA, b=22.0
alter 4nx4 and resid 218 and name CA, b=17.0
alter 4nx4 and resid 219 and name CA, b=16.0
alter 4nx4 and resid 220 and name CA, b=18.0
alter 4nx4 and resid 221 and name CA, b=25.0
alter 4nx4 and resid 222 and name CA, b=32.0
alter 4nx4 and resid 223 and name CA, b=28.0
alter 4nx4 and resid 224 and name CA, b=24.0
alter 4nx4 and resid 225 and name CA, b=19.0
alter 4nx4 and resid 226 and name CA, b=28.0
alter 4nx4 and resid 227 and name CA, b=25.0
alter 4nx4 and resid 228 and name CA, b=13.0
alter 4nx4 and resid 229 and name CA, b=17.0
alter 4nx4 and resid 230 and name CA, b=21.0
alter 4nx4 and resid 231 and name CA, b=23.0
alter 4nx4 and resid 232 and name CA, b=15.0
alter 4nx4 and resid 233 and name CA, b=17.0
alter 4nx4 and resid 234 and name CA, b=29.0
alter 4nx4 and resid 235 and name CA, b=35.0
alter 4nx4 and resid 236 and name CA, b=31.0
alter 4nx4 and resid 237 and name CA, b=28.0
alter 4nx4 and resid 238 and name CA, b=32.0
alter 4nx4 and resid 239 and name CA, b=27.0
alter 4nx4 and resid 240 and name CA, b=14.0
alter 4nx4 and resid 241 and name CA, b=14.0
alter 4nx4 and resid 242 and name CA, b=12.0
alter 4nx4 and resid 243 and name CA, b=19.0
alter 4nx4 and resid 244 and name CA, b=11.0
alter 4nx4 and resid 245 and name CA, b=9.0
alter 4nx4 and resid 246 and name CA, b=13.0
alter 4nx4 and resid 247 and name CA, b=18.0
alter 4nx4 and resid 248 and name CA, b=17.0
alter 4nx4 and resid 249 and name CA, b=1.0
alter 4nx4 and resid 250 and name CA, b=2.0
alter 4nx4 and resid 251 and name CA, b=12.0
alter 4nx4 and resid 252 and name CA, b=6.0
alter 4nx4 and resid 253 and name CA, b=7.0
alter 4nx4 and resid 254 and name CA, b=14.0
alter 4nx4 and resid 255 and name CA, b=2.0
alter 4nx4 and resid 256 and name CA, b=10.0
alter 4nx4 and resid 257 and name CA, b=22.0
alter 4nx4 and resid 258 and name CA, b=35.0
alter 4nx4 and resid 259 and name CA, b=24.0
alter 4nx4 and resid 260 and name CA, b=22.0
alter 4nx4 and resid 261 and name CA, b=23.0
alter 4nx4 and resid 262 and name CA, b=32.0
alter 4nx4 and resid 263 and name CA, b=22.0
alter 4nx4 and resid 264 and name CA, b=23.0
alter 4nx4 and resid 265 and name CA, b=13.0
alter 4nx4 and resid 266 and name CA, b=14.0
alter 4nx4 and resid 267 and name CA, b=1.0
alter 4nx4 and resid 268 and name CA, b=9.0
alter 4nx4 and resid 269 and name CA, b=17.0
alter 4nx4 and resid 270 and name CA, b=23.0
alter 4nx4 and resid 271 and name CA, b=15.0
alter 4nx4 and resid 272 and name CA, b=24.0
alter 4nx4 and resid 273 and name CA, b=23.0
alter 4nx4 and resid 274 and name CA, b=38.0
alter 4nx4 and resid 275 and name CA, b=22.0
alter 4nx4 and resid 276 and name CA, b=18.0
alter 4nx4 and resid 277 and name CA, b=10.0
alter 4nx4 and resid 278 and name CA, b=20.0
alter 4nx4 and resid 279 and name CA, b=0.0
alter 4nx4 and resid 280 and name CA, b=14.0
alter 4nx4 and resid 281 and name CA, b=34.0
alter 4nx4 and resid 282 and name CA, b=36.0
alter 4nx4 and resid 283 and name CA, b=29.0
alter 4nx4 and resid 284 and name CA, b=33.0
alter 4nx4 and resid 285 and name CA, b=47.0
alter 4nx4 and resid 286 and name CA, b=25.0
alter 4nx4 and resid 287 and name CA, b=11.0
alter 4nx4 and resid 288 and name CA, b=22.0
alter 4nx4 and resid 289 and name CA, b=32.0
alter 4nx4 and resid 290 and name CA, b=29.0
alter 4nx4 and resid 291 and name CA, b=25.0
alter 4nx4 and resid 292 and name CA, b=23.0
alter 4nx4 and resid 293 and name CA, b=28.0
alter 4nx4 and resid 294 and name CA, b=28.0
alter 4nx4 and resid 295 and name CA, b=23.0
alter 4nx4 and resid 296 and name CA, b=24.0
alter 4nx4 and resid 297 and name CA, b=30.0
alter 4nx4 and resid 298 and name CA, b=29.0
alter 4nx4 and resid 299 and name CA, b=25.0
alter 4nx4 and resid 300 and name CA, b=23.0
alter 4nx4 and resid 301 and name CA, b=22.0
alter 4nx4 and resid 302 and name CA, b=21.0
alter 4nx4 and resid 303 and name CA, b=18.0
alter 4nx4 and resid 304 and name CA, b=28.0
alter 4nx4 and resid 305 and name CA, b=16.0
alter 4nx4 and resid 306 and name CA, b=16.0
alter 4nx4 and resid 307 and name CA, b=25.0
alter 4nx4 and resid 308 and name CA, b=23.0
alter 4nx4 and resid 309 and name CA, b=14.0
alter 4nx4 and resid 310 and name CA, b=10.0
alter 4nx4 and resid 311 and name CA, b=19.0
alter 4nx4 and resid 312 and name CA, b=30.0
alter 4nx4 and resid 313 and name CA, b=28.0
alter 4nx4 and resid 314 and name CA, b=22.0
alter 4nx4 and resid 315 and name CA, b=29.0
alter 4nx4 and resid 316 and name CA, b=31.0
alter 4nx4 and resid 317 and name CA, b=35.0
alter 4nx4 and resid 318 and name CA, b=26.0
alter 4nx4 and resid 319 and name CA, b=17.0
alter 4nx4 and resid 320 and name CA, b=16.0
alter 4nx4 and resid 321 and name CA, b=17.0
alter 4nx4 and resid 322 and name CA, b=17.0
alter 4nx4 and resid 323 and name CA, b=14.0
alter 4nx4 and resid 324 and name CA, b=11.0
alter 4nx4 and resid 325 and name CA, b=18.0
alter 4nx4 and resid 326 and name CA, b=0.0
alter 4nx4 and resid 327 and name CA, b=2.0
alter 4nx4 and resid 328 and name CA, b=10.0
alter 4nx4 and resid 329 and name CA, b=8.0
alter 4nx4 and resid 330 and name CA, b=18.0
alter 4nx4 and resid 331 and name CA, b=31.0
alter 4nx4 and resid 332 and name CA, b=29.0
alter 4nx4 and resid 333 and name CA, b=21.0
alter 4nx4 and resid 334 and name CA, b=24.0
alter 4nx4 and resid 335 and name CA, b=28.0
alter 4nx4 and resid 336 and name CA, b=28.0
alter 4nx4 and resid 337 and name CA, b=22.0
alter 4nx4 and resid 338 and name CA, b=24.0
alter 4nx4 and resid 339 and name CA, b=32.0
alter 4nx4 and resid 340 and name CA, b=28.0
alter 4nx4 and resid 341 and name CA, b=17.0
alter 4nx4 and resid 342 and name CA, b=15.0
alter 4nx4 and resid 343 and name CA, b=20.0
alter 4nx4 and resid 344 and name CA, b=17.0
alter 4nx4 and resid 345 and name CA, b=26.0
alter 4nx4 and resid 346 and name CA, b=7.0
alter 4nx4 and resid 347 and name CA, b=8.0
alter 4nx4 and resid 348 and name CA, b=18.0
alter 4nx4 and resid 349 and name CA, b=10.0
alter 4nx4 and resid 350 and name CA, b=12.0
alter 4nx4 and resid 351 and name CA, b=3.0
alter 4nx4 and resid 352 and name CA, b=3.0
sort
rebuild

# Set background color
bg_color white
