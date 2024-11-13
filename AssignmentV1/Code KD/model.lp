\ Model MarketClearingProblem1
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  13.32 Gen_DABid[0,0] + 13.32 Gen_DABid[1,0] + 20.7 Gen_DABid[2,0]
   + 20.93 Gen_DABid[3,0] + 26.11 Gen_DABid[4,0] + 10.52 Gen_DABid[5,0]
   + 10.52 Gen_DABid[6,0] + 6.02 Gen_DABid[7,0] + 5.47 Gen_DABid[8,0]
   + 10.52 Gen_DABid[10,0] + 10.89 Gen_DABid[11,0] + 40 Gen_DABid[13,0]
   + 5 Gen_DABid[14,0] + 20 Gen_DABid[15,0] - 1000 Dem_DABid[0,0]
   - 125 Dem_DABid[1,0] - 75 Dem_DABid[2,0] - 1000 Dem_DABid[3,0]
   - 125 Dem_DABid[4,0] - 75 Dem_DABid[5,0] - 1000 Dem_DABid[6,0]
   - 125 Dem_DABid[7,0] - 75 Dem_DABid[8,0] - 1000 Dem_DABid[9,0]
   - 125 Dem_DABid[10,0] - 75 Dem_DABid[11,0] - 1000 Dem_DABid[12,0]
   - 125 Dem_DABid[13,0] - 75 Dem_DABid[14,0] - 1000 Dem_DABid[15,0]
   - 125 Dem_DABid[16,0]
Subject To
 Production_Capacity_Constraint_(0,_0): Gen_DABid[0,0] <= 152
 Production_Capacity_Constraint_(1,_0): Gen_DABid[1,0] <= 152
 Production_Capacity_Constraint_(2,_0): Gen_DABid[2,0] <= 350
 Production_Capacity_Constraint_(3,_0): Gen_DABid[3,0] <= 591
 Production_Capacity_Constraint_(4,_0): Gen_DABid[4,0] <= 40
 Production_Capacity_Constraint_(5,_0): Gen_DABid[5,0] <= 155
 Production_Capacity_Constraint_(6,_0): Gen_DABid[6,0] <= 155
 Production_Capacity_Constraint_(7,_0): Gen_DABid[7,0] <= 200
 Production_Capacity_Constraint_(8,_0): Gen_DABid[8,0] <= 200
 Production_Capacity_Constraint_(9,_0): Gen_DABid[9,0] <= 300
 Production_Capacity_Constraint_(10,_0): Gen_DABid[10,0] <= 310
 Production_Capacity_Constraint_(11,_0): Gen_DABid[11,0] <= 350
 Production_Capacity_Constraint_(12,_0): Gen_DABid[12,0] <= 0
 Production_Capacity_Constraint_(13,_0): Gen_DABid[13,0] <= 0
 Production_Capacity_Constraint_(14,_0): Gen_DABid[14,0] <= 0
 Production_Capacity_Constraint_(15,_0): Gen_DABid[15,0] <= 0
 Demand_Capacity_Constraint_(0,_0): Dem_DABid[0,0] <= 49.59459892841733
 Demand_Capacity_Constraint_(1,_0): Dem_DABid[1,0] <= 60.37839
 Demand_Capacity_Constraint_(2,_0): Dem_DABid[2,0] <= 0
 Demand_Capacity_Constraint_(3,_0): Dem_DABid[3,0] <= 33.93314663523292
 Demand_Capacity_Constraint_(4,_0): Dem_DABid[4,0] <= 44.395875
 Demand_Capacity_Constraint_(5,_0): Dem_DABid[5,0] <= 0
 Demand_Capacity_Constraint_(6,_0): Dem_DABid[6,0] <= 57.42532507500955
 Demand_Capacity_Constraint_(7,_0): Dem_DABid[7,0] <= 106.5501
 Demand_Capacity_Constraint_(8,_0): Dem_DABid[8,0] <= 0
 Demand_Capacity_Constraint_(9,_0): Dem_DABid[9,0] <= 88.74822966137839
 Demand_Capacity_Constraint_(10,_0): Dem_DABid[10,0] <= 165.152655
 Demand_Capacity_Constraint_(11,_0): Dem_DABid[11,0] <= 0
 Demand_Capacity_Constraint_(12,_0): Dem_DABid[12,0] <= 144.8684337119559
 Demand_Capacity_Constraint_(13,_0): Dem_DABid[13,0] <= 62.154225
 Demand_Capacity_Constraint_(14,_0): Dem_DABid[14,0] <= 0
 Demand_Capacity_Constraint_(15,_0): Dem_DABid[15,0] <= 83.52774556365024
 Demand_Capacity_Constraint_(16,_0): Dem_DABid[16,0] <= 79.912575
 Power_Balance_Constraint_(0,_0): Gen_DABid[0,0] + Gen_DABid[1,0]
   + Gen_DABid[2,0] + Gen_DABid[3,0] + Gen_DABid[4,0] + Gen_DABid[5,0]
   + Gen_DABid[6,0] + Gen_DABid[12,0] + Gen_DABid[14,0] - Dem_DABid[0,0]
   - Dem_DABid[2,0] - Dem_DABid[4,0] - Dem_DABid[6,0] - Dem_DABid[8,0]
   - Dem_DABid[10,0] - Dem_DABid[12,0] - Dem_DABid[14,0] - Dem_DABid[16,0]
   = 0
 Power_Balance_Constraint_(1,_0): Gen_DABid[7,0] + Gen_DABid[8,0]
   + Gen_DABid[9,0] + Gen_DABid[10,0] + Gen_DABid[11,0] + Gen_DABid[13,0]
   + Gen_DABid[15,0] - Dem_DABid[1,0] - Dem_DABid[3,0] - Dem_DABid[5,0]
   - Dem_DABid[7,0] - Dem_DABid[9,0] - Dem_DABid[11,0] - Dem_DABid[13,0]
   - Dem_DABid[15,0] = 0
Bounds
End
