import os
import numpy as np
import traceback
# from commonroad_utils.parser.scenario import get_scenario
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.common.file_reader import CommonRoadFileReader


path = os.getcwd() + '/data/demo/FOP_Scenarios/'
save_dir = os.getcwd() + '/data/FOP_Scenarios_transformed/'

for sc in os.listdir(path):
      if sc.endswith(".xml"):
            # scenario, planning_problem, pp_set = get_scenario(os.path.join(path, dir), sc)
            scenario, planning_problem_set = CommonRoadFileReader(os.path.join(path, sc)).open()
            planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
            start_pos = planning_problem.initial_state.position

            scenario.translate_rotate(translation=np.array([-start_pos[0], -start_pos[1]]), angle=0)
            planning_problem.translate_rotate(translation=np.array([-start_pos[0], -start_pos[1]]), angle=0)

            scenario.lanelet_network._buffered_polygons = {
            id: lane.polygon.shapely_object
            for id, lane in scenario.lanelet_network._lanelets.items() }
            
            scenario.lanelet_network._create_strtree()

            try:
                  route_planner = RoutePlanner(scenario, 
                                          planning_problem)
                   
                                         

                  route = route_planner.plan_routes().retrieve_first_route()
                  
            except Exception as e:
                  print("Error in ", sc)
                  traceback.print_exc()
                  continue
            
            if route:
                  fw = CommonRoadFileWriter(
                                    scenario,
                                    planning_problem_set,
                                    scenario.author,
                                    scenario.affiliation,
                                    scenario.source,
                                    scenario.tags
                                    )
                              
                  fw.write_to_file(os.path.join(save_dir, sc), OverwriteExistingFile.ALWAYS)
                  
                  print("Saved ", sc)
                  
print("Done!")