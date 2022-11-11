#!python
#cython: language_level=3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
cimport numpy as np
cimport cython

import deepnovo_config

mass_ID_np = deepnovo_config.mass_ID_np
cdef int GO_ID = deepnovo_config.GO_ID
cdef int EOS_ID = deepnovo_config.EOS_ID
cdef float mass_H2O = deepnovo_config.mass_H2O
cdef float mass_NH3 = deepnovo_config.mass_NH3
cdef float mass_H = deepnovo_config.mass_H
cdef float mass_CO = deepnovo_config.mass_CO
cdef int WINDOW_SIZE = deepnovo_config.WINDOW_SIZE
cdef int vocab_size = deepnovo_config.vocab_size
cdef int num_ion = deepnovo_config.num_ion


def get_sinusoid_encoding_table(n_position, embed_size, padding_idx=0):
    """ Sinusoid position encoding table
    n_position: maximum integer that the embedding op could receive
    embed_size: embed size
    return
      a embedding matrix of shape [n_position, embed_size]
    """

    def cal_angle(position, hid_idx):
        return position / np.power(deepnovo_config.sinusoid_base, 2 * (hid_idx // 2) / embed_size)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(embed_size)]

    sinusoid_matrix = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position + 1)], dtype=np.float32)

    sinusoid_matrix[:, 0::2] = np.sin(sinusoid_matrix[:, 0::2])  # dim 2i
    sinusoid_matrix[:, 1::2] = np.cos(sinusoid_matrix[:, 1::2])  # dim 2i+1

    sinusoid_matrix[padding_idx] = 0.
    return sinusoid_matrix

sinusoid_matrix = get_sinusoid_encoding_table(deepnovo_config.n_position, deepnovo_config.embedding_size,
                                              padding_idx=deepnovo_config.PAD_ID)

@cython.boundscheck(False) # turn off bounds-checking
@cython.wraparound(False) # turn off negative index wrapping
def get_ion_index(peptide_mass, prefix_mass, direction):
  """

  :param peptide_mass: neutral mass of a peptide
  :param prefix_mass:
  :param direction: 0 for forward, 1 for backward
  :return: an int32 ndarray of shape [26, 8], each element represent a index of the spectrum embbeding matrix. for out
  of bound position, the index is 0
  """
  if direction == 0:
    candidate_b_mass = prefix_mass + mass_ID_np
    candidate_y_mass = peptide_mass - candidate_b_mass
  elif direction == 1:
    candidate_y_mass = prefix_mass + mass_ID_np
    candidate_b_mass = peptide_mass - candidate_y_mass
  candidate_a_mass = candidate_b_mass - mass_CO

  mass_A = [71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711, 71.03711]
  mass_A1 = candidate_b_mass + mass_A
  mass_A2 = candidate_y_mass + mass_A
  mass_A3 = candidate_a_mass + mass_A
  mass_R = [156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111, 156.10111]
  mass_R1 = candidate_b_mass + mass_R
  mass_R2 = candidate_y_mass + mass_R
  mass_R3 = candidate_a_mass + mass_R
  mass_N = [114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293, 114.04293]
  mass_N1 = candidate_b_mass + mass_N
  mass_N2 = candidate_y_mass + mass_N
  mass_N3 = candidate_a_mass + mass_N
  mass_Nd = [115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695, 115.02695]
  mass_Nd1 = candidate_b_mass + mass_Nd
  mass_Nd2 = candidate_y_mass + mass_Nd
  mass_Nd3 = candidate_a_mass + mass_Nd
  mass_D = [115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694, 115.02694]
  mass_D1 = candidate_b_mass + mass_D
  mass_D2 = candidate_y_mass + mass_D
  mass_D3 = candidate_a_mass + mass_D
  mass_Cc = [160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065, 160.03065]
  mass_Cc1 = candidate_b_mass + mass_Cc
  mass_Cc2 = candidate_y_mass + mass_Cc
  mass_Cc3 = candidate_a_mass + mass_Cc
  mass_E = [129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259, 129.04259]
  mass_E1 = candidate_b_mass + mass_E
  mass_E2 = candidate_y_mass + mass_E
  mass_E3 = candidate_a_mass + mass_E
  mass_Q = [128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858, 128.05858]
  mass_Q1 = candidate_b_mass + mass_Q
  mass_Q2 = candidate_y_mass + mass_Q
  mass_Q3 = candidate_a_mass + mass_Q
  mass_Qd = [129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426, 129.0426]
  mass_Qd1 = candidate_b_mass + mass_Qd
  mass_Qd2 = candidate_y_mass + mass_Qd
  mass_Qd3 = candidate_a_mass + mass_Qd
  mass_G = [57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146, 57.02146]
  mass_G1 = candidate_b_mass + mass_G
  mass_G2 = candidate_y_mass + mass_G
  mass_G3 = candidate_a_mass + mass_G
  mass_HH = [137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891, 137.05891]
  mass_HH1 = candidate_b_mass + mass_HH
  mass_HH2 = candidate_y_mass + mass_HH
  mass_HH3 = candidate_a_mass + mass_HH
  mass_II = [113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406]
  mass_II1 = candidate_b_mass + mass_II
  mass_II2= candidate_y_mass + mass_II
  mass_II3 = candidate_a_mass + mass_II
  mass_L = [113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406, 113.08406]
  mass_L1 = candidate_b_mass + mass_L
  mass_L2 = candidate_y_mass + mass_L
  mass_L3 = candidate_a_mass + mass_L
  mass_K = [128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496, 128.09496]
  mass_K1 = candidate_b_mass + mass_K
  mass_K2 = candidate_y_mass + mass_K
  mass_K3 = candidate_a_mass + mass_K
  mass_M = [131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049, 131.04049]
  mass_M1 = candidate_b_mass + mass_M
  mass_M2 = candidate_y_mass + mass_M
  mass_M3 = candidate_a_mass + mass_M
  mass_Mo = [147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354, 147.0354]
  mass_Mo1 = candidate_b_mass + mass_Mo
  mass_Mo2 = candidate_y_mass + mass_Mo
  mass_Mo3 = candidate_a_mass + mass_Mo
  mass_F = [147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841, 147.06841]
  mass_F1 = candidate_b_mass + mass_F
  mass_F2 = candidate_y_mass + mass_F
  mass_F3 = candidate_a_mass + mass_F
  mass_P = [97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276, 97.05276]
  mass_P1 = candidate_b_mass + mass_P
  mass_P2 = candidate_y_mass + mass_P
  mass_P3 = candidate_a_mass + mass_P
  mass_S = [87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203, 87.03203]
  mass_S1 = candidate_b_mass + mass_S
  mass_S2 = candidate_y_mass + mass_S
  mass_S3 = candidate_a_mass + mass_S
  mass_T = [101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768, 101.04768]
  mass_T1 = candidate_b_mass + mass_T
  mass_T2 = candidate_y_mass + mass_T
  mass_T3 = candidate_a_mass + mass_T
  mass_W = [186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931, 186.07931]
  mass_W1 = candidate_b_mass + mass_W
  mass_W2 = candidate_y_mass + mass_W
  mass_W3 = candidate_a_mass + mass_W
  mass_Y = [163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333, 163.06333]
  mass_Y1 = candidate_b_mass + mass_Y
  mass_Y2 = candidate_y_mass + mass_Y
  mass_Y3 = candidate_a_mass + mass_Y
  mass_V = [99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841, 99.06841]
  mass_V1 = candidate_b_mass + mass_V
  mass_V2 = candidate_y_mass + mass_V
  mass_V3 = candidate_a_mass + mass_V

  # b-ions
  candidate_b_H2O = candidate_b_mass - mass_H2O
  candidate_b_NH3 = candidate_b_mass - mass_NH3
  candidate_b_plus2_charge1 = ((candidate_b_mass + 2 * mass_H) / 2
                               - mass_H)

  # a-ions
  candidate_a_H2O = candidate_a_mass - mass_H2O
  candidate_a_NH3 = candidate_a_mass - mass_NH3
  candidate_a_plus2_charge1 = ((candidate_a_mass + 2 * mass_H) / 2
                               - mass_H)

  # y-ions
  candidate_y_H2O = candidate_y_mass - mass_H2O
  candidate_y_NH3 = candidate_y_mass - mass_NH3
  candidate_y_plus2_charge1 = ((candidate_y_mass + 2 * mass_H) / 2
                               - mass_H)

  # ion_2
  # ~   b_ions = [candidate_b_mass]
  # ~   y_ions = [candidate_y_mass]
  # ~   ion_mass_list = b_ions + y_ions

  # ion_8
  b_ions = [candidate_b_mass,
            mass_A1,
            mass_R1,
            mass_N1,
            mass_Nd1,
            mass_D1,
            mass_Cc1,
            mass_E1,
            mass_Q1,
            mass_Qd1,
            mass_G1,
            mass_HH1,
            mass_II1,
            mass_L1,
            mass_K1,
            mass_M1,
            mass_Mo1,
            mass_F1,
            mass_P1,
            mass_S1,
            mass_T1,
            mass_W1,
            mass_Y1,
            mass_V1,
            candidate_b_H2O,
            candidate_b_NH3,
            candidate_b_plus2_charge1]
  y_ions = [candidate_y_mass,
            mass_A2,
            mass_R2,
            mass_N2,
            mass_Nd2,
            mass_D2,
            mass_Cc2,
            mass_E2,
            mass_Q2,
            mass_Qd2,
            mass_G2,
            mass_HH2,
            mass_II2,
            mass_L2,
            mass_K2,
            mass_M2,
            mass_Mo2,
            mass_F2,
            mass_P2,
            mass_S2,
            mass_T2,
            mass_W2,
            mass_Y2,
            mass_V2,
            candidate_y_H2O,
            candidate_y_NH3,
            candidate_y_plus2_charge1]
  a_ions = [candidate_a_mass,
            candidate_a_H2O,
            candidate_a_NH3,
            candidate_a_plus2_charge1]
  ion_mass_list = b_ions + y_ions + a_ions
  ion_mass = np.array(ion_mass_list, dtype=np.float32)  # 8 by 26

  # ion locations
  # ion_location = np.ceil(ion_mass * SPECTRUM_RESOLUTION).astype(np.int64) # 8 by 26

  in_bound_mask = np.logical_and(
      ion_mass > 0,
      ion_mass <= deepnovo_config.MZ_MAX).astype(np.float32)
  ion_location = ion_mass * in_bound_mask  # 8 by 26, out of bound index would have value 0
  return ion_location.transpose()  # 26 by 8


def pad_to_length(data: list, length, pad_token=0.):
  """
  pad data to length if len(data) is smaller than length
  :param data:
  :param length:
  :param pad_token:
  :return:
  """
  for i in range(length - len(data)):
    data.append(pad_token)


def process_peaks(spectrum_mz_list, spectrum_intensity_list, peptide_mass):
  """

  :param spectrum_mz_list:
  :param spectrum_intensity_list:
  :param peptide_mass: peptide neutral mass
  :return:
    peak_location: int64, [N]
    peak_intensity: float32, [N]
    spectrum_representation: float32 [embedding_size]
  """
  charge = 1.0
  spectrum_intensity_max = np.max(spectrum_intensity_list)
  # charge 1 peptide location
  spectrum_mz_list.append(peptide_mass + charge*deepnovo_config.mass_H)
  spectrum_intensity_list.append(spectrum_intensity_max)

  # N-terminal, b-ion, peptide_mass_C
  # append N-terminal
  mass_N = deepnovo_config.mass_N_terminus - deepnovo_config.mass_H
  spectrum_mz_list.append(mass_N + charge*deepnovo_config.mass_H)
  spectrum_intensity_list.append(spectrum_intensity_max)
  # append peptide_mass_C
  mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
  peptide_mass_C = peptide_mass - mass_C
  spectrum_mz_list.append(peptide_mass_C + charge*deepnovo_config.mass_H)
  spectrum_intensity_list.append(spectrum_intensity_max)

  # C-terminal, y-ion, peptide_mass_N
  # append C-terminal
  mass_C = deepnovo_config.mass_C_terminus + deepnovo_config.mass_H
  spectrum_mz_list.append(mass_C + charge*deepnovo_config.mass_H)
  spectrum_intensity_list.append(spectrum_intensity_max)


  pad_to_length(spectrum_mz_list, deepnovo_config.MAX_NUM_PEAK)
  pad_to_length(spectrum_intensity_list, deepnovo_config.MAX_NUM_PEAK)

  spectrum_mz = np.array(spectrum_mz_list, dtype=np.float32)
  spectrum_mz_location = np.ceil(spectrum_mz * deepnovo_config.spectrum_reso).astype(np.int32)

  neutral_mass = spectrum_mz - charge*deepnovo_config.mass_H
  in_bound_mask = np.logical_and(neutral_mass > 0., neutral_mass < deepnovo_config.MZ_MAX)
  neutral_mass[~in_bound_mask] = 0.
  # intensity
  spectrum_intensity = np.array(spectrum_intensity_list, dtype=np.float32)
  norm_intensity = spectrum_intensity / spectrum_intensity_max

  spectrum_representation = np.zeros(deepnovo_config.embedding_size, dtype=np.float32)
  for i, loc in enumerate(spectrum_mz_location):
    if loc < 0.5 or loc > deepnovo_config.n_position:
      continue
    else:
      spectrum_representation += sinusoid_matrix[loc] * norm_intensity[i]

  top_N_indices = np.argpartition(norm_intensity, -deepnovo_config.MAX_NUM_PEAK)[-deepnovo_config.MAX_NUM_PEAK:]
  intensity = norm_intensity[top_N_indices]
  mass_location = neutral_mass[top_N_indices]

  return mass_location, intensity, spectrum_representation
