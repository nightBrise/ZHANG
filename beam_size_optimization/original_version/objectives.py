# class EMIT_PRF29(ObjBase):
    
#     # img_pv = 'LA-BI:PRF29:IMG:ArrayData'
#     img_pv = 'LA-BI:PRF29:RAW:ArrayData'
#     quad = 'LA-PS:Q49:SETI'
#     gain_pv = 'LA-BI:PRF29:CAM:GainRaw'
#     beam_not_at_Linac = Beam_not_at_Linac_end()
    
#     def _objective(self):
        
#         size = 0
#         for current, gain in zip([-0.8, 0.0, 0.8], [500, 500, 500]):
#             caput(self.gain_pv, gain)
#             caput(self.quad, current)
#             time.sleep(0.4)
#             img = 0
#             for _ in range(4):
                
                
#                 time.sleep(0.21)
#                 while not self.stop_flag.token and (spark() or self.beam_not_at_Linac()) and caget('IN-MW:KLY3:GET_INTERLOCK_STATE')==1:
#                     print('spark')
#                     time.sleep(5)
                
#                 img = caget(self.img_pv).reshape((1392, 1040), order="F")
#                 size_= np.array(get_size_emit(img))

#                 loop_ = 0
#                 while np.any(size_>500) and loop_<3:
#                     time.sleep(0.25)
#                     img = caget(self.img_pv).reshape((1392, 1040), order="F")
#                     size_= np.array(get_size_emit(img))
#                     loop_ += 1

#                 print(size_)
#                 size_[size_<0.3] = 1.5
            
#                 size += np.sum(size_)
            
#         caput(self.gain_pv, 0)
#         caput(self.quad, 0.0)
#         return size/24
