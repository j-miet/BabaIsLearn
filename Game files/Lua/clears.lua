--[[
All the code in this file is from official clear.lua file and is written and provided by the developer!
Only change here is to wrap print on line 68 in comments to prevent unnecessary outputs. 
--]]

function clearunits(restore_)
	units = {}
	tiledunits = {}
	codeunits = {}
	animunits = {}
	unitlists = {}
	undobuffer = {}
	unitmap = {}
	unittypeshere = {}
	prevunitmap = {}
	ruleids = {}
	objectlist = {}
	objectpalette = {}
	updatelist = {}
	objectcolours = {}
	wordunits = {}
	wordrelatedunits = {}
	letterunits = {}
	letterunits_map = {}
	paths = {}
	paradox = {}
	movelist = {}
	deleted = {}
	effecthistory = {}
	notfeatures = {}
	groupfeatures = {}
	groupmembers = {}
	pushedunits = {}
	customobjects = {}
	cobjects = {}
	condstatus = {}
	emptydata = {}
	leveldata = {}
	leveldata.colours = {}
	leveldata.currcolour = 0
	poweredstatus = {}
	specialtiling = {}
	
	visiontargets = {}
	vision_rendered = {}
	
	generaldata.values[CURRID] = 1
	updateundo = true
	hiddenmap = nil
	levelconversions = {}
	last_key = 0
	auto_dir = {}
	destroylevel_check = false
	destroylevel_style = ""
	
	HACK_MOVES = 0
	HACK_INFINITY = 0
	movemap = {}
	
	local restore = true
	if (restore_ ~= nil) then
		restore = norestore_
	end
	
	if restore then
		newundo()
		
		--print("clearunits")   --disable to debloat saved output
		
		restoredefaults()
	end
end