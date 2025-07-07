--[[ Modpack for Baba Is you that allows fetching basic game data through console prints.

Guides to understand game modding:
https://github.com/PlasmaFlare/baba-modding-guide
https://github.com/cg5-/baba-modding-guide

<<How to use>>
-create a modpack folder inside 'Baba Is You/Data/Worlds'. Folder name shouldn't matter, but it could be 'babaislearn'
-copy the files inside Game Files folder to 'Baba Is You/Data/Worlds/babaislearn'
-game disables os & io libraries so only way to get data out is to enable printing into console, then redirect this 
 output into elsewhere. Files above enable text printing. To get data, here's a few options:

	> open powershell -> cd to Baba Is You steam folder -> type the command 
		& '.\Baba Is You.exe' | echo

	> alternatively, run git bash -> cmd -> cd to Baba folder and type the command
		"Baba Is You.exe" | cat

    > third option, which is used in this program: open game inside a Python subprocess, direct subprocess output into
      subprocess stdout pipe.

Values (these can be found under constants.lue in Baba Is You game folder):
-units contain all non-empty level objects
 >unit.strings[UNITNAME] -> get name of detailed name of object; prefix 'text_' if text object
 >unit.values[XPOS], unit.values[YPOS] -> x and y locations
-level size ROOMSIZEX, ROOMSIZEY. These counts also borders, which add 2 to each, so just subtract this value off.
 Max size without modding the values is 33x18.
-RESTARTED displays true/false when restart is pressed.
-NOPLAYER displays 0 if something is You, otherwise 1.

Modding api has support for hook functions; these enable data printing after a specific event occurs.
]]--

function print_mapdata()
	for _, v in pairs(units) do
		print(v.strings[UNITNAME] ..','.. v.values[XPOS] .. ',' .. v.values[YPOS] .. '+')
	end
	--print("_noplayer:"..generaldata2.values[NOPLAYER])  RE-ENABLE THIS IF PLAYER STATUS IS NEEDED
end

table.insert(mod_hook_functions["turn_end"],
    function ()
        print("------")
        print_mapdata()
        print("------")
    end
)

table.insert(mod_hook_functions["undoed_after"],
    function ()
        print("------")
        print_mapdata()
        print("------")
    end
)

table.insert(mod_hook_functions["level_start"],
	function ()
        print("_size:"..tostring(generaldata.values[ROOMSIZEX]-2)..","..tostring(generaldata.values[ROOMSIZEY]-2))
        print("------")
        print_mapdata()
        print("------")
    end
)

table.insert(mod_hook_functions["level_win"],
	function()
        print("###")
        print_mapdata()
        print("###")
	end
)

--[[
table.insert(mod_hook_functions["level_restart"],
	function()
        print("------")
		print("RESTARTED: "..tostring(generaldata.flags[RESTARTED]))
	end
)
]]--