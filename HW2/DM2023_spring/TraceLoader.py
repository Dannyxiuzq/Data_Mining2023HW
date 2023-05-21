"""
author:yqtong@buaa.edu.cn
date:2023-05-04
"""
from utils import *


class ObjectTrace:
    def __init__(self):
        self.radar_dict = {}
        self.protection_unit_dict = {}
        self.air_defence_dict = {}
        self.ship_dict = {}
        self.missile_launcher_dict = {}
        self.hist_trace_dict = {}
        self.hist_action_dict = {}
        self.timestamp_index = 0
        self.id_index = 1
        self.unitname_index = 2
        self.purpose_index = 3
        self.aircraft_type_index = 4
        # 编队索引
        self.formation_index = 5
        # 敌我识别索引
        self.coalition_index = 6
        # 纬度索引
        self.latitude_index = 7
        # 经度索引
        self.longitude_index = 8
        # 高度索引
        self.altitude_index = 9
        # 航向索引
        self.heading_index = 10
        # 俯仰索引
        self.pitch_index = 11
        # 滚转索引
        self.bank_index = 12
        self.vx_index = 13
        self.vy_index = 14
        self.vz_index = 15
        self.v_index = 16
        self.interfere_flag = 17
        self.group = 18

    def parser_line(self, planes):
        self.group2id = {}
        self.group_index = []
        self.id2vel = {}
        self.id2xyz = {}
        self.id2group = {}
        self.unitname2vel = {}
        self.unitname2xyz = {}
        self.formation2num = {}
        for idx, line_list in enumerate(planes):
            timestamp = line_list[self.timestamp_index]
            Id = line_list[self.id_index]
            unitname = line_list[self.unitname_index]
            intention = line_list[self.purpose_index]
            aircraft_model = line_list[self.aircraft_type_index]
            formation = line_list[self.formation_index]
            if aircraft_model == '' or aircraft_model == 'Pilot':
                continue
            aircraft_type = type_dict[aircraft_model]
            coalition = line_list[self.coalition_index]
            latitude, longitude, altitude = float(line_list[self.latitude_index]),\
                                            float(line_list[self.longitude_index]),\
                                            float(line_list[self.altitude_index])
            heading, pitch, bank = float(line_list[self.heading_index]),\
                                   float(line_list[self.pitch_index]),\
                                   float(line_list[self.bank_index])
            vx, vy, vz, v = float(line_list[self.vx_index]),\
                            float(line_list[self.vy_index]),\
                            float(line_list[self.vz_index]),\
                            float(line_list[self.v_index])
            group = int(line_list[self.group])

            if intention == '干扰':
                interfere_flag = True
            else:
                interfere_flag = False

            xyz = lla2xyz([latitude, longitude, altitude])

            if aircraft_type == 'Radar':
                self.radar_dict[Id] = {
                    'Timestamp': [timestamp],
                    'Group': group,
                    'Model': aircraft_model,
                    'Type': aircraft_type,
                    'IFF': coalition,
                    'lla': [[latitude, longitude, altitude]],
                    'xyz': [xyz],
                    'formation': [formation],
                }
            elif aircraft_type == 'ProtectionUnit':
                self.protection_unit_dict[Id] = {
                    'Timestamp': [timestamp],
                    'Group': group,
                    'Model': aircraft_model,
                    'Type': aircraft_type,
                    'IFF': coalition,
                    'lla': [[latitude, longitude, altitude]],
                    'xyz': [xyz],
                    'formation': [formation],
                }
            elif aircraft_type == 'AirDefence':
                self.air_defence_dict[Id] = {
                    'Timestamp': [timestamp],
                    'Group': group,
                    'Model': aircraft_model,
                    'Type': aircraft_type,
                    'IFF': coalition,
                    'lla': [[latitude, longitude, altitude]],
                    'xyz': [xyz],
                    'formation': [formation],
                }
            elif aircraft_type == 'MissileLauncher' or aircraft_type == 'BallisticMissileLauncher':
                self.missile_launcher_dict[Id] = {
                    'Timestamp': [timestamp],
                    'Group': group,
                    'Model': aircraft_model,
                    'Type': aircraft_type,
                    'IFF': coalition,
                    'lla': [[latitude, longitude, altitude]],
                    'xyz': [xyz],
                    'formation': [formation],
                }
            elif aircraft_type == 'Warship':
                self.ship_dict[Id] = {
                    'Timestamp': [timestamp],
                    'Group': group,
                    'Model': aircraft_model,
                    'Type': aircraft_type,
                    'IFF': coalition,
                    'lla': [[latitude, longitude, altitude]],
                    'xyz': [xyz],
                    'formation': [formation],
                }
            else:
                self.id2vel[Id] = [vx, vy, vz]
                self.id2xyz[Id] = xyz
                self.id2group[Id] = unitname
                self.unitname2vel[unitname] = [vx, vy, vz]
                self.unitname2xyz[unitname] = xyz
                self.group2id[unitname] = Id
                self.group_index.append(Id)
                unitname_list = unitname.split('-')
                within_formation_id = unitname_list[-1]
                if len(unitname_list) == 3:
                    formation_id = unitname_list[0] + '-单机-' + unitname_list[1]
                elif len(unitname_list) == 4:
                    formation_id = unitname_list[0] + '-' + unitname_list[1] + '-' + unitname_list[2]
                else:
                    formation_id = Id + within_formation_id

                if not self.formation2num.__contains__(formation_id):
                    self.formation2num[formation_id] = 1
                else:
                    self.formation2num[formation_id] += 1

                if not self.hist_trace_dict.__contains__(Id):
                    self.hist_trace_dict[Id] = {
                        'Timestamp': [timestamp],
                        'Unitname': unitname,
                        'Group': group,
                        'FormationId': formation_id,
                        'Model': aircraft_model,
                        'Type': aircraft_type,
                        'IFF': coalition,
                        'lla': [[latitude, longitude, altitude]],
                        'xyz': [xyz],
                        'Delta_h': [],
                        'Att': [[heading, pitch, bank]],
                        'Vel': [[vx, vy, vz]],
                        'Speed': [v],
                        'Delta_v': [],
                        'Delta_att': [],
                        'Interfere': [interfere_flag],
                        'Intention': [intention],
                        'Intention_pred': [],
                        'Points_num': 1,
                        'Formation': [formation],
                        'Formation_pred': [],
                        'Formation_num_pred': [],
                        'Wingman': [],
                    }
                else:
                    self.hist_trace_dict[Id]['Timestamp'].append(timestamp)
                    pre_height = self.hist_trace_dict[Id]['xyz'][-1][-1]
                    self.hist_trace_dict[Id]['Delta_h'].append(altitude - pre_height)
                    self.hist_trace_dict[Id]['lla'].append([latitude, longitude, altitude])
                    self.hist_trace_dict[Id]['xyz'].append(xyz)
                    pre_heading, pre_pitch, pre_bank = self.hist_trace_dict[Id]['Att'][-1][0],\
                                                       self.hist_trace_dict[Id]['Att'][-1][1],\
                                                       self.hist_trace_dict[Id]['Att'][-1][2]
                    self.hist_trace_dict[Id]['Delta_att'].append([
                        heading - pre_heading,
                        pitch - pre_pitch,
                        bank - pre_bank
                    ])
                    self.hist_trace_dict[Id]['Att'].append([
                        heading, pitch, bank
                    ])
                    self.hist_trace_dict[Id]['Vel'].append([vx, vy, vz])
                    pre_speed = self.hist_trace_dict[Id]['Speed'][-1]
                    self.hist_trace_dict[Id]['Delta_v'].append(v - pre_speed)
                    self.hist_trace_dict[Id]['Speed'].append(v)
                    self.hist_trace_dict[Id]['Interfere'].append(interfere_flag)
                    self.hist_trace_dict[Id]['Points_num'] += 1