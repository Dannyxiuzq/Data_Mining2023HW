"""
author:yqtong@buaa.edu.cn
date:2023-05-04
"""
import math


# 将不同的地面、空中型号与类别对应
type_dict = {
    'SNR_75V': 'Radar',
    'SON_9': 'Radar',
    'RPC_5N62V': 'Radar',
    'SA-11 Buk CC 9S470M1': 'ProtectionUnit',
    'ZIL-135': 'ProtectionUnit',
    'KrAZ6322': 'ProtectionUnit',
    'S_75M_Volhov': 'AirDefence',
    'Ural-4320 APA-5D': 'ProtectionUnit',
    'ATMZ-5': 'ProtectionUnit',
    'ZSU-23-4 Shilka': 'AirDefence',
    'RLS_19J6': 'Radar',
    'ATZ-5': 'ProtectionUnit',
    'p-19 s-125 sr': 'Radar',
    'F-4E': 'Fixed-wing',
    'E-2C': 'Fixed-wing',
    'MQ-9 Reaper': 'Fixed-wing',
    'RQ-1A Predator': 'Fixed-wing',
    'KC130': 'Fixed-wing',
    'CH-53E': 'Helicopter',
    'BGM_109B': 'BGM',
    'AGM_45': 'AGM',
    'AGM_45A': 'AGM',
    'TICONDEROG': 'Warship',
    'Scud_B': 'BallisticMissileLauncher',
    'SCUD_RAKETA': 'BallisticMissile',
    'Silkworm_SR': 'Radar',
    'SKP-11': 'ProtectionUnit',
    'Stennis': 'BallisticMissileLauncher',
    'F-16A': 'Fixed-wing',
}

intention_type = ['侦察', '打击', '诱扰', '指挥', '干扰']
protection_type = ['ProtectionUnit', 'Radar', 'AirDefence', 'Warship', 'MissileLauncher', 'BallisticMissileLauncher']

# 地球信息
PI = 3.14159265358979323846 # 圆周率近似值
EPSILON = 0.000000000001
D2R = PI / 180
R2D = 180 / PI
EARTH_R = 6378137.0
EARTH_OBLATENESS_inverse = 298.257223563
EARTH_R_SHORT = EARTH_R - (EARTH_R / EARTH_OBLATENESS_inverse)
EARTH_ECCENTRICITY = math.sqrt(EARTH_R * EARTH_R - EARTH_R_SHORT * EARTH_R_SHORT) / EARTH_R


def lla2xyz(trace):
    """
    坐标系变换,将经度、纬度和高度（LLA）转换为笛卡尔坐标系（XYZ）,细节不用关注
    :param trace:
    :return:
    """
    lat, lon, alt = trace[0], trace[1], trace[2]
    L = lon * D2R
    B = lat * D2R
    H = alt
    N = EARTH_R / math.sqrt(1 - EARTH_ECCENTRICITY * EARTH_ECCENTRICITY * math.sin(B) * math.sin(B))
    # 根据笛卡尔坐标系的转换公式计算出对应的xyz坐标
    x = (N + H) * math.cos(B) * math.cos(L)
    y = (N + H) * math.cos(B) * math.sin(L)
    z = (N * (1 - EARTH_ECCENTRICITY * EARTH_ECCENTRICITY) + H) * math.sin(B)
    return [x, y, z]