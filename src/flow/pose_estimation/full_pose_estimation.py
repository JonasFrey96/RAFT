from .flow_to_trafo_PnP import flow_to_trafo_PnP
from loss import AddSLoss
import torch
import numpy as np
from .pose_estimate_violations import Violation

def full_pose_estimation(
	h_gt, 
	h_render,
	h_init,
	bb, 
	flow_valid, 
	flow_pred, 
	idx,
	K_ren,
	K_real,
	render_d,
	model_points, 
	cfg = {}
	):
	"""
	
	Returns ADD-S for each element in batch!
	"""

	real_tl, real_br, ren_tl, ren_br = bb
	typ = flow_pred.dtype
	
	h_pred = h_init.clone()
	ratios = []
	repro_errors = []


	sucs = []
	violations = []
	for b in range( real_tl.shape[0] ):
		suc, h_pre, repro_error, ratio, violation = flow_to_trafo_PnP( 
			real_br = real_br[b].type(typ).to('cpu'), 
			real_tl = real_tl[b].type(typ).to('cpu'), 
			ren_br = ren_br[b].type(typ).to('cpu'), 
			ren_tl = ren_tl[b].type(typ).to('cpu'), 
			flow_mask = (flow_valid[b][:,:].to('cpu') == 1), 
			u_map = flow_pred[b][ 0, :, :].to('cpu'), 
			v_map = flow_pred[b][ 1, :, :].to('cpu'), 
			K_ren = K_ren[b].type(typ).to('cpu'), 
			K_real = K_real[b].to('cpu'), 
			render_d = render_d[b].to('cpu'), 
			h_render = h_render[b].to('cpu').clone(),
			h_real_est = h_init[b].to('cpu').clone(),
			** cfg.get("flow_to_trafo_PnP", {})
			)
		sucs.append(suc)
		h_pred[b,:,:] = h_pre.clone().to(h_init.device)
		ratios.append(ratio)
		repro_errors.append( repro_error )
		violations.append(violation)


	h_pred.to('cuda')
	h_gt = h_gt.type(torch.float32)
	target = torch.bmm ( model_points, torch.transpose(h_gt[:,:3,:3], 1,2 )  ) + h_gt[:,:3,3][:,None,:].repeat(1, model_points.shape[1],1)

	typ = torch.float32
	h_pred = h_pred.type(typ).to('cuda')
	h_gt = h_gt.type(typ).to('cuda')
	h_init = h_init.type(typ).to('cuda')
	
	adds = AddSLoss( sym_list = list( range(0,22)))
	add_s = AddSLoss( sym_list = [] ) # bowl, wood_block, large clamp, extra_large clamp, foam_brick

	
	add_s_h_pred = add_s( target.clone(), model_points.clone(), idx, H=h_pred)
	# if ( adds_h_pred > 0.04).sum() > 0:
		# print("ratio", ratios, "adds_h_pred", adds_h_pred)
	add_s_h_init = add_s( target.clone(), model_points.clone(), idx, H=h_init) 
	add_s_h_gt = add_s( target.clone(), model_points.clone(), idx, H=h_gt)

	adds_h_pred = adds( target.clone(), model_points.clone(), idx, H=h_pred)
	# if ( adds_h_pred > 0.04).sum() > 0:
		# print("ratio", ratios, "adds_h_pred", adds_h_pred)
	adds_h_init = adds( target.clone(), model_points.clone(), idx, H=h_init) 
	adds_h_gt = adds( target.clone(), model_points.clone(), idx, H=h_gt)

	for i, ratio in enumerate( ratios):
		if ratio < cfg.get('inlier_ratio_constraint', 0):
			violations[i] = Violation.INLIER_RATIO_CONSTRAINT

	limit = torch.norm( h_pred[:,:3,3] - h_init[:,:3,3], dim=1, p=2) > cfg.get('trust_region_constraint', 0.1)

	if limit.sum() == real_tl.shape[0]:
		for i in range( real_tl.shape[0] ):
			if violations[i] != Violation.SUCCESS:
				violations[i] = Violation.TRUST_REGION_CONSTRAINT
		
		print( "Failed NORM: ", torch.norm( h_pred[:,:3,3] - h_init[:,:3,3], dim=1, p=2), "ADDs" ,adds_h_pred  )
	
	valid = limit == False
	count_invalid = real_tl.shape[0]- valid.sum()

	return {"adds_h_pred": adds_h_pred,
					"adds_h_init": adds_h_init,
					"add_s_h_pred": add_s_h_pred,
					"add_s_h_init": add_s_h_init,
					}, count_invalid, h_pred, np.array( repro_errors), np.array( ratios ),  valid, violations