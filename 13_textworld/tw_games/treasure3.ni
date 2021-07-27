Use MAX_STATIC_DATA of 500000.
When play begins, seed the random-number generator with 1234.

container is a kind of thing.
door is a kind of thing.
object-like is a kind of thing.
supporter is a kind of thing.
food is a kind of object-like.
key is a kind of object-like.
containers are openable, lockable and fixed in place. containers are usually closed.
door is openable and lockable.
object-like is portable.
supporters are fixed in place.
food is edible.
A room has a text called internal name.


The r_0 and the r_2 and the r_5 and the r_3 and the r_7 and the r_1 and the r_8 and the r_9 and the r_4 and the r_6 are rooms.

Understand "studio" as r_0.
The internal name of r_0 is "studio".
The printed name of r_0 is "-= Studio =-".
The studio part 0 is some text that varies. The studio part 0 is "You're now in the studio.

 You can see [if c_0 is locked]a locked[else if c_0 is open]an opened[otherwise]a closed[end if]".
The studio part 1 is some text that varies. The studio part 1 is " box in the room.[if c_0 is open and there is something in the c_0] The box contains [a list of things in the c_0].[end if]".
The studio part 2 is some text that varies. The studio part 2 is "[if c_0 is open and the c_0 contains nothing] The box is empty! What a waste of a day![end if]".
The studio part 3 is some text that varies. The studio part 3 is " You make out a toolbox.[if c_1 is open and there is something in the c_1] The toolbox contains [a list of things in the c_1].[end if]".
The studio part 4 is some text that varies. The studio part 4 is "[if c_1 is open and the c_1 contains nothing] The toolbox is empty, what a horrible day![end if]".
The studio part 5 is some text that varies. The studio part 5 is " Look over there! a table. [if there is something on the s_0]On the table you make out [a list of things on the s_0].[end if]".
The studio part 6 is some text that varies. The studio part 6 is "[if there is nothing on the s_0]However, the table, like an empty table, has nothing on it.[end if]".
The studio part 7 is some text that varies. The studio part 7 is " You can see a mantle. The mantle is ordinary.[if there is something on the s_1] On the mantle you can make out [a list of things on the s_1].[end if]".
The studio part 8 is some text that varies. The studio part 8 is "[if there is nothing on the s_1] However, the mantle, like an empty mantle, has nothing on it. You swear loudly.[end if]".
The studio part 9 is some text that varies. The studio part 9 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The studio part 10 is some text that varies. The studio part 10 is " gateway leading north. There is [if d_3 is open]an open[otherwise]a closed[end if]".
The studio part 11 is some text that varies. The studio part 11 is " portal leading south. There is [if d_5 is open]an open[otherwise]a closed[end if]".
The studio part 12 is some text that varies. The studio part 12 is " door leading west.".
The description of r_0 is "[studio part 0][studio part 1][studio part 2][studio part 3][studio part 4][studio part 5][studio part 6][studio part 7][studio part 8][studio part 9][studio part 10][studio part 11][studio part 12]".

west of r_0 and east of r_2 is a door called d_5.
south of r_0 and north of r_5 is a door called d_3.
north of r_0 and south of r_1 is a door called d_0.
Understand "shower" as r_2.
The internal name of r_2 is "shower".
The printed name of r_2 is "-= Shower =-".
The shower part 0 is some text that varies. The shower part 0 is "You are in a shower. An ordinary one. The room seems oddly familiar, as though it were only superficially different from the other rooms in the building.



 There is [if d_5 is open]an open[otherwise]a closed[end if]".
The shower part 1 is some text that varies. The shower part 1 is " door leading east. There is [if d_4 is open]an open[otherwise]a closed[end if]".
The shower part 2 is some text that varies. The shower part 2 is " passageway leading south. There is an unguarded exit to the north.".
The description of r_2 is "[shower part 0][shower part 1][shower part 2]".

south of r_2 and north of r_3 is a door called d_4.
The r_4 is mapped north of r_2.
east of r_2 and west of r_0 is a door called d_5.
Understand "workshop" as r_5.
The internal name of r_5 is "workshop".
The printed name of r_5 is "-= Workshop =-".
The workshop part 0 is some text that varies. The workshop part 0 is "You've entered a workshop. You decide to just list off a complete list of everything you see in the room, because hey, why not?



 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The workshop part 1 is some text that varies. The workshop part 1 is " gate leading east. There is [if d_3 is open]an open[otherwise]a closed[end if]".
The workshop part 2 is some text that varies. The workshop part 2 is " portal leading north. There is [if d_2 is open]an open[otherwise]a closed[end if]".
The workshop part 3 is some text that varies. The workshop part 3 is " hatch leading west.".
The description of r_5 is "[workshop part 0][workshop part 1][workshop part 2][workshop part 3]".

west of r_5 and east of r_3 is a door called d_2.
north of r_5 and south of r_0 is a door called d_3.
east of r_5 and west of r_8 is a door called d_1.
Understand "basement" as r_3.
The internal name of r_3 is "basement".
The printed name of r_3 is "-= Basement =-".
The basement part 0 is some text that varies. The basement part 0 is "Ah, the basement. This is some kind of basement, really great standard vibes in this place, a wonderful standard atmosphere. You start to take note of what's in the room.



 There is [if d_2 is open]an open[otherwise]a closed[end if]".
The basement part 1 is some text that varies. The basement part 1 is " hatch leading east. There is [if d_4 is open]an open[otherwise]a closed[end if]".
The basement part 2 is some text that varies. The basement part 2 is " passageway leading north. There is an unguarded exit to the south.".
The description of r_3 is "[basement part 0][basement part 1][basement part 2]".

The r_6 is mapped south of r_3.
north of r_3 and south of r_2 is a door called d_4.
east of r_3 and west of r_5 is a door called d_2.
Understand "garage" as r_7.
The internal name of r_7 is "garage".
The printed name of r_7 is "-= Garage =-".
The garage part 0 is some text that varies. The garage part 0 is "Well, here we are in the garage.

 You see a rack. Now why would someone leave that there? The rack is typical.[if there is something on the s_2] On the rack you can see [a list of things on the s_2].[end if]".
The garage part 1 is some text that varies. The garage part 1 is "[if there is nothing on the s_2] But the thing is empty.[end if]".
The garage part 2 is some text that varies. The garage part 2 is "

You don't like doors? Why not try going west, that entranceway is unblocked.".
The description of r_7 is "[garage part 0][garage part 1][garage part 2]".

The r_1 is mapped west of r_7.
Understand "cookery" as r_1.
The internal name of r_1 is "cookery".
The printed name of r_1 is "-= Cookery =-".
The cookery part 0 is some text that varies. The cookery part 0 is "You find yourself in a cookery. A standard kind of place.

 You see a platter. The platter is usual.[if there is something on the s_3] On the platter you see [a list of things on the s_3].[end if]".
The cookery part 1 is some text that varies. The cookery part 1 is "[if there is nothing on the s_3] Looks like someone's already been here and taken everything off it, though.[end if]".
The cookery part 2 is some text that varies. The cookery part 2 is "

 There is [if d_0 is open]an open[otherwise]a closed[end if]".
The cookery part 3 is some text that varies. The cookery part 3 is " gateway leading south. You don't like doors? Why not try going east, that entranceway is unblocked.".
The description of r_1 is "[cookery part 0][cookery part 1][cookery part 2][cookery part 3]".

south of r_1 and north of r_0 is a door called d_0.
The r_7 is mapped east of r_1.
Understand "kitchenette" as r_8.
The internal name of r_8 is "kitchenette".
The printed name of r_8 is "-= Kitchenette =-".
The kitchenette part 0 is some text that varies. The kitchenette part 0 is "You're now in a kitchenette.



 There is [if d_1 is open]an open[otherwise]a closed[end if]".
The kitchenette part 1 is some text that varies. The kitchenette part 1 is " gate leading west. You need an unblocked exit? You should try going east.".
The description of r_8 is "[kitchenette part 0][kitchenette part 1]".

west of r_8 and east of r_5 is a door called d_1.
The r_9 is mapped east of r_8.
Understand "bedchamber" as r_9.
The internal name of r_9 is "bedchamber".
The printed name of r_9 is "-= Bedchamber =-".
The bedchamber part 0 is some text that varies. The bedchamber part 0 is "You are in a bedchamber. An ordinary kind of place.

 You can make out a safe. The light flickers for a second, but nothing else happens.[if c_2 is open and there is something in the c_2] The safe contains [a list of things in the c_2].[end if]".
The bedchamber part 1 is some text that varies. The bedchamber part 1 is "[if c_2 is open and the c_2 contains nothing] The safe is empty, what a horrible day![end if]".
The bedchamber part 2 is some text that varies. The bedchamber part 2 is "

You don't like doors? Why not try going west, that entranceway is unguarded.".
The description of r_9 is "[bedchamber part 0][bedchamber part 1][bedchamber part 2]".

The r_8 is mapped west of r_9.
Understand "laundromat" as r_4.
The internal name of r_4 is "laundromat".
The printed name of r_4 is "-= Laundromat =-".
The laundromat part 0 is some text that varies. The laundromat part 0 is "You are in a laundromat. A standard one.

 You make out a shelf. [if there is something on the s_4]On the shelf you see [a list of things on the s_4].[end if]".
The laundromat part 1 is some text that varies. The laundromat part 1 is "[if there is nothing on the s_4]Unfortunately, there isn't a thing on it.[end if]".
The laundromat part 2 is some text that varies. The laundromat part 2 is " You make out a board. [if there is something on the s_5]On the board you make out [a list of things on the s_5], so there's that.[end if]".
The laundromat part 3 is some text that varies. The laundromat part 3 is "[if there is nothing on the s_5]But the thing is empty.[end if]".
The laundromat part 4 is some text that varies. The laundromat part 4 is "

There is an exit to the south. Don't worry, it is unblocked.".
The description of r_4 is "[laundromat part 0][laundromat part 1][laundromat part 2][laundromat part 3][laundromat part 4]".

The r_2 is mapped south of r_4.
Understand "washroom" as r_6.
The internal name of r_6 is "washroom".
The printed name of r_6 is "-= Washroom =-".
The washroom part 0 is some text that varies. The washroom part 0 is "You've entered a washroom. I guess you better just go and list everything you see here.

 You can see [if c_3 is locked]a locked[else if c_3 is open]an opened[otherwise]a closed[end if]".
The washroom part 1 is some text that varies. The washroom part 1 is " dresser right there by you.[if c_3 is open and there is something in the c_3] The dresser contains [a list of things in the c_3].[end if]".
The washroom part 2 is some text that varies. The washroom part 2 is "[if c_3 is open and the c_3 contains nothing] Empty! What kind of nightmare TextWorld is this?[end if]".
The washroom part 3 is some text that varies. The washroom part 3 is "

There is an exit to the north. Don't worry, it is unblocked.".
The description of r_6 is "[washroom part 0][washroom part 1][washroom part 2][washroom part 3]".

The r_3 is mapped north of r_6.

The c_0 and the c_1 and the c_2 and the c_3 are containers.
The c_0 and the c_1 and the c_2 and the c_3 are privately-named.
The d_0 and the d_3 and the d_5 and the d_4 and the d_2 and the d_1 are doors.
The d_0 and the d_3 and the d_5 and the d_4 and the d_2 and the d_1 are privately-named.
The f_0 are foods.
The f_0 are privately-named.
The o_0 are object-likes.
The o_0 are privately-named.
The r_0 and the r_2 and the r_5 and the r_3 and the r_7 and the r_1 and the r_8 and the r_9 and the r_4 and the r_6 are rooms.
The r_0 and the r_2 and the r_5 and the r_3 and the r_7 and the r_1 and the r_8 and the r_9 and the r_4 and the r_6 are privately-named.
The s_0 and the s_1 and the s_2 and the s_3 and the s_4 and the s_5 are supporters.
The s_0 and the s_1 and the s_2 and the s_3 and the s_4 and the s_5 are privately-named.

The description of d_0 is "it is what it is, a gateway [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_0 is "gateway".
Understand "gateway" as d_0.
The d_0 is open.
The description of d_3 is "it's a rugged portal [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_3 is "portal".
Understand "portal" as d_3.
The d_3 is open.
The description of d_5 is "it is what it is, a door [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_5 is "door".
Understand "door" as d_5.
The d_5 is open.
The description of d_4 is "it's a noble passageway [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_4 is "passageway".
Understand "passageway" as d_4.
The d_4 is open.
The description of d_2 is "it is what it is, a hatch [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of d_2 is "hatch".
Understand "hatch" as d_2.
The d_2 is open.
The description of d_1 is "The gate looks stuffy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of d_1 is "gate".
Understand "gate" as d_1.
The d_1 is open.
The description of c_0 is "The box looks strong, and impossible to break. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_0 is "box".
Understand "box" as c_0.
The c_0 is in r_0.
The c_0 is locked.
The description of c_1 is "The toolbox looks strong, and impossible to destroy. [if open]You can see inside it.[else if closed]You can't see inside it because the lid's in your way.[otherwise]There is a lock on it.[end if]".
The printed name of c_1 is "toolbox".
Understand "toolbox" as c_1.
The c_1 is in r_0.
The c_1 is locked.
The description of c_2 is "The safe looks strong, and impossible to crack. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_2 is "safe".
Understand "safe" as c_2.
The c_2 is in r_9.
The c_2 is locked.
The description of c_3 is "The dresser looks strong, and impossible to break. [if open]It is open.[else if closed]It is closed.[otherwise]It is locked.[end if]".
The printed name of c_3 is "dresser".
Understand "dresser" as c_3.
The c_3 is in r_6.
The c_3 is locked.
The description of f_0 is "You couldn't pay me to eat that usual thing.".
The printed name of f_0 is "garlic clove".
Understand "garlic clove" as f_0.
Understand "garlic" as f_0.
Understand "clove" as f_0.
The f_0 is in r_9.
The description of o_0 is "The cane looks to fit in here".
The printed name of o_0 is "cane".
Understand "cane" as o_0.
The o_0 is in r_7.
The description of s_0 is "The table is an unstable piece of junk.".
The printed name of s_0 is "table".
Understand "table" as s_0.
The s_0 is in r_0.
The description of s_1 is "The mantle is undependable.".
The printed name of s_1 is "mantle".
Understand "mantle" as s_1.
The s_1 is in r_0.
The description of s_2 is "The rack is shaky.".
The printed name of s_2 is "rack".
Understand "rack" as s_2.
The s_2 is in r_7.
The description of s_3 is "The platter is wobbly.".
The printed name of s_3 is "platter".
Understand "platter" as s_3.
The s_3 is in r_1.
The description of s_4 is "The shelf is solid.".
The printed name of s_4 is "shelf".
Understand "shelf" as s_4.
The s_4 is in r_4.
The description of s_5 is "The board is reliable.".
The printed name of s_5 is "board".
Understand "board" as s_5.
The s_5 is in r_4.


The player is in r_0.

The quest0 completed is a truth state that varies.
The quest0 completed is usually false.

Test quest0_0 with "go north / go east / take cane"

Every turn:
	if quest0 completed is true:
		do nothing;
	else if The player carries the f_0:
		end the story; [Lost]
	else if The player is in r_7 and The player carries the o_0:
		increase the score by 1; [Quest completed]
		Now the quest0 completed is true;

Use scoring. The maximum score is 1.
This is the simpler notify score changes rule:
	If the score is not the last notified score:
		let V be the score - the last notified score;
		say "Your score has just gone up by [V in words] ";
		if V > 1:
			say "points.";
		else:
			say "point.";
		Now the last notified score is the score;
	if score is maximum score:
		end the story finally; [Win]

The simpler notify score changes rule substitutes for the notify score changes rule.

Rule for listing nondescript items:
	stop.

Rule for printing the banner text:
	say "[fixed letter spacing]";
	say "                    ________  ________  __    __  ________        [line break]";
	say "                   |        \|        \|  \  |  \|        \       [line break]";
	say "                    \$$$$$$$$| $$$$$$$$| $$  | $$ \$$$$$$$$       [line break]";
	say "                      | $$   | $$__     \$$\/  $$   | $$          [line break]";
	say "                      | $$   | $$  \     >$$  $$    | $$          [line break]";
	say "                      | $$   | $$$$$    /  $$$$\    | $$          [line break]";
	say "                      | $$   | $$_____ |  $$ \$$\   | $$          [line break]";
	say "                      | $$   | $$     \| $$  | $$   | $$          [line break]";
	say "                       \$$    \$$$$$$$$ \$$   \$$    \$$          [line break]";
	say "              __       __   ______   _______   __        _______  [line break]";
	say "             |  \  _  |  \ /      \ |       \ |  \      |       \ [line break]";
	say "             | $$ / \ | $$|  $$$$$$\| $$$$$$$\| $$      | $$$$$$$\[line break]";
	say "             | $$/  $\| $$| $$  | $$| $$__| $$| $$      | $$  | $$[line break]";
	say "             | $$  $$$\ $$| $$  | $$| $$    $$| $$      | $$  | $$[line break]";
	say "             | $$ $$\$$\$$| $$  | $$| $$$$$$$\| $$      | $$  | $$[line break]";
	say "             | $$$$  \$$$$| $$__/ $$| $$  | $$| $$_____ | $$__/ $$[line break]";
	say "             | $$$    \$$$ \$$    $$| $$  | $$| $$     \| $$    $$[line break]";
	say "              \$$      \$$  \$$$$$$  \$$   \$$ \$$$$$$$$ \$$$$$$$ [line break]";
	say "[variable letter spacing][line break]";
	say "[objective][line break]".

Include Basic Screen Effects by Emily Short.

Rule for printing the player's obituary:
	if story has ended finally:
		center "*** The End ***";
	else:
		center "*** You lost! ***";
	say paragraph break;
	say "You scored [score] out of a possible [maximum score], in [turn count] turn(s).";
	[wait for any key;
	stop game abruptly;]
	rule succeeds.

Rule for implicitly taking something (called target):
	if target is fixed in place:
		say "The [target] is fixed in place.";
	otherwise:
		say "You need to take the [target] first.";
		set pronouns from target;
	stop.

Does the player mean doing something:
	if the noun is not nothing and the second noun is nothing and the player's command matches the text printed name of the noun:
		it is likely;
	if the noun is nothing and the second noun is not nothing and the player's command matches the text printed name of the second noun:
		it is likely;
	if the noun is not nothing and the second noun is not nothing and the player's command matches the text printed name of the noun and the player's command matches the text printed name of the second noun:
		it is very likely.  [Handle action with two arguments.]

Printing the content of the room is an activity.
Rule for printing the content of the room:
	let R be the location of the player;
	say "Room contents:[line break]";
	list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the world is an activity.
Rule for printing the content of the world:
	let L be the list of the rooms;
	say "World: [line break]";
	repeat with R running through L:
		say "  [the internal name of R][line break]";
	repeat with R running through L:
		say "[the internal name of R]:[line break]";
		if the list of things in R is empty:
			say "  nothing[line break]";
		otherwise:
			list the contents of R, with newlines, indented, including all contents, with extra indentation.

Printing the content of the inventory is an activity.
Rule for printing the content of the inventory:
	say "You are carrying: ";
	list the contents of the player, as a sentence, giving inventory information, including all contents;
	say ".".

The print standard inventory rule is not listed in any rulebook.
Carry out taking inventory (this is the new print inventory rule):
	say "You are carrying: ";
	list the contents of the player, as a sentence, giving inventory information, including all contents;
	say ".".

Printing the content of nowhere is an activity.
Rule for printing the content of nowhere:
	say "Nowhere:[line break]";
	let L be the list of the off-stage things;
	repeat with thing running through L:
		say "  [thing][line break]";

Printing the things on the floor is an activity.
Rule for printing the things on the floor:
	let R be the location of the player;
	let L be the list of things in R;
	remove yourself from L;
	remove the list of containers from L;
	remove the list of supporters from L;
	remove the list of doors from L;
	if the number of entries in L is greater than 0:
		say "There is [L with indefinite articles] on the floor.";

After printing the name of something (called target) while
printing the content of the room
or printing the content of the world
or printing the content of the inventory
or printing the content of nowhere:
	follow the property-aggregation rules for the target.

The property-aggregation rules are an object-based rulebook.
The property-aggregation rulebook has a list of text called the tagline.

[At the moment, we only support "open/unlocked", "closed/unlocked" and "closed/locked" for doors and containers.]
[A first property-aggregation rule for an openable open thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable closed thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an lockable unlocked thing (this is the mention unlocked lockable rule):
	add "unlocked" to the tagline.

A property-aggregation rule for an lockable locked thing (this is the mention locked lockable rule):
	add "locked" to the tagline.]

A first property-aggregation rule for an openable lockable open unlocked thing (this is the mention open openables rule):
	add "open" to the tagline.

A property-aggregation rule for an openable lockable closed unlocked thing (this is the mention closed openables rule):
	add "closed" to the tagline.

A property-aggregation rule for an openable lockable closed locked thing (this is the mention locked openables rule):
	add "locked" to the tagline.

A property-aggregation rule for a lockable thing (called the lockable thing) (this is the mention matching key of lockable rule):
	let X be the matching key of the lockable thing;
	if X is not nothing:
		add "match [X]" to the tagline.

A property-aggregation rule for an edible off-stage thing (this is the mention eaten edible rule):
	add "eaten" to the tagline.

The last property-aggregation rule (this is the print aggregated properties rule):
	if the number of entries in the tagline is greater than 0:
		say " ([tagline])";
		rule succeeds;
	rule fails;

The objective part 0 is some text that varies. The objective part 0 is "Get ready to pick stuff up and put it in places, because you've just entered TextWorld! Here is your task for today. Your first objective is to head north. Okay, and then, travel east. Then, recover t".
The objective part 1 is some text that varies. The objective part 1 is "he cane from the floor of the garage. Once that's all handled, you can stop!".

An objective is some text that varies. The objective is "[objective part 0][objective part 1]".
Printing the objective is an action applying to nothing.
Carry out printing the objective:
	say "[objective]".

Understand "goal" as printing the objective.

The taking action has an object called previous locale (matched as "from").

Setting action variables for taking:
	now previous locale is the holder of the noun.

Report taking something from the location:
	say "You pick up [the noun] from the ground." instead.

Report taking something:
	say "You take [the noun] from [the previous locale]." instead.

Report dropping something:
	say "You drop [the noun] on the ground." instead.

The print state option is a truth state that varies.
The print state option is usually false.

Turning on the print state option is an action applying to nothing.
Carry out turning on the print state option:
	Now the print state option is true.

Turning off the print state option is an action applying to nothing.
Carry out turning off the print state option:
	Now the print state option is false.

Printing the state is an activity.
Rule for printing the state:
	let R be the location of the player;
	say "Room: [line break] [the internal name of R][line break]";
	[say "[line break]";
	carry out the printing the content of the room activity;]
	say "[line break]";
	carry out the printing the content of the world activity;
	say "[line break]";
	carry out the printing the content of the inventory activity;
	say "[line break]";
	carry out the printing the content of nowhere activity;
	say "[line break]".

Printing the entire state is an action applying to nothing.
Carry out printing the entire state:
	say "-=STATE START=-[line break]";
	carry out the printing the state activity;
	say "[line break]Score:[line break] [score]/[maximum score][line break]";
	say "[line break]Objective:[line break] [objective][line break]";
	say "[line break]Inventory description:[line break]";
	say "  You are carrying: [a list of things carried by the player].[line break]";
	say "[line break]Room description:[line break]";
	try looking;
	say "[line break]-=STATE STOP=-";

Every turn:
	if extra description command option is true:
		say "<description>";
		try looking;
		say "</description>";
	if extra inventory command option is true:
		say "<inventory>";
		try taking inventory;
		say "</inventory>";
	if extra score command option is true:
		say "<score>[line break][score][line break]</score>";
	if extra score command option is true:
		say "<moves>[line break][turn count][line break]</moves>";
	if print state option is true:
		try printing the entire state;

When play ends:
	if print state option is true:
		try printing the entire state;

After looking:
	carry out the printing the things on the floor activity.

Understand "print_state" as printing the entire state.
Understand "enable print state option" as turning on the print state option.
Understand "disable print state option" as turning off the print state option.

Before going through a closed door (called the blocking door):
	say "You have to open the [blocking door] first.";
	stop.

Before opening a locked door (called the locked door):
	let X be the matching key of the locked door;
	if X is nothing:
		say "The [locked door] is welded shut.";
	otherwise:
		say "You have to unlock the [locked door] with the [X] first.";
	stop.

Before opening a locked container (called the locked container):
	let X be the matching key of the locked container;
	if X is nothing:
		say "The [locked container] is welded shut.";
	otherwise:
		say "You have to unlock the [locked container] with the [X] first.";
	stop.

Displaying help message is an action applying to nothing.
Carry out displaying help message:
	say "[fixed letter spacing]Available commands:[line break]";
	say "  look:                describe the current room[line break]";
	say "  goal:                print the goal of this game[line break]";
	say "  inventory:           print player's inventory[line break]";
	say "  go <dir>:            move the player north, east, south or west[line break]";
	say "  examine ...:         examine something more closely[line break]";
	say "  eat ...:             eat edible food[line break]";
	say "  open ...:            open a door or a container[line break]";
	say "  close ...:           close a door or a container[line break]";
	say "  drop ...:            drop an object on the floor[line break]";
	say "  take ...:            take an object that is on the floor[line break]";
	say "  put ... on ...:      place an object on a supporter[line break]";
	say "  take ... from ...:   take an object from a container or a supporter[line break]";
	say "  insert ... into ...: place an object into a container[line break]";
	say "  lock ... with ...:   lock a door or a container with a key[line break]";
	say "  unlock ... with ...: unlock a door or a container with a key[line break]";

Understand "help" as displaying help message.

Taking all is an action applying to nothing.
Check taking all:
	say "You have to be more specific!";
	rule fails.

Understand "take all" as taking all.
Understand "get all" as taking all.
Understand "pick up all" as taking all.

Understand "take each" as taking all.
Understand "get each" as taking all.
Understand "pick up each" as taking all.

Understand "take everything" as taking all.
Understand "get everything" as taking all.
Understand "pick up everything" as taking all.

The extra description command option is a truth state that varies.
The extra description command option is usually false.

Turning on the extra description command option is an action applying to nothing.
Carry out turning on the extra description command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra description command option is true.

Understand "tw-extra-infos description" as turning on the extra description command option.

The extra inventory command option is a truth state that varies.
The extra inventory command option is usually false.

Turning on the extra inventory command option is an action applying to nothing.
Carry out turning on the extra inventory command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra inventory command option is true.

Understand "tw-extra-infos inventory" as turning on the extra inventory command option.

The extra score command option is a truth state that varies.
The extra score command option is usually false.

Turning on the extra score command option is an action applying to nothing.
Carry out turning on the extra score command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra score command option is true.

Understand "tw-extra-infos score" as turning on the extra score command option.

The extra moves command option is a truth state that varies.
The extra moves command option is usually false.

Turning on the extra moves command option is an action applying to nothing.
Carry out turning on the extra moves command option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the extra moves command option is true.

Understand "tw-extra-infos moves" as turning on the extra moves command option.

To trace the actions:
	(- trace_actions = 1; -).

Tracing the actions is an action applying to nothing.
Carry out tracing the actions:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	trace the actions;

Understand "tw-trace-actions" as tracing the actions.

The restrict commands option is a truth state that varies.
The restrict commands option is usually false.

Turning on the restrict commands option is an action applying to nothing.
Carry out turning on the restrict commands option:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	Now the restrict commands option is true.

Understand "restrict commands" as turning on the restrict commands option.

The taking allowed flag is a truth state that varies.
The taking allowed flag is usually false.

Before removing something from something:
	now the taking allowed flag is true.

After removing something from something:
	now the taking allowed flag is false.

Before taking a thing (called the object) when the object is on a supporter (called the supporter):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [supporter] instead.";
		rule fails.

Before of taking a thing (called the object) when the object is in a container (called the container):
	if the restrict commands option is true and taking allowed flag is false:
		say "Can't see any [object] on the floor! Try taking the [object] from the [container] instead.";
		rule fails.

Understand "take [something]" as removing it from.

Rule for supplying a missing second noun while removing:
	if restrict commands option is false and noun is on a supporter (called the supporter):
		now the second noun is the supporter;
	else if restrict commands option is false and noun is in a container (called the container):
		now the second noun is the container;
	else:
		try taking the noun;
		say ""; [Needed to avoid printing a default message.]

The version number is always 1.

Reporting the version number is an action applying to nothing.
Carry out reporting the version number:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	say "[version number]".

Understand "tw-print version" as reporting the version number.

Reporting max score is an action applying to nothing.
Carry out reporting max score:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	say "[maximum score]".

Understand "tw-print max_score" as reporting max score.

To print id of (something - thing):
	(- print {something}, "^"; -).

Printing the id of player is an action applying to nothing.
Carry out printing the id of player:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of player.

Printing the id of EndOfObject is an action applying to nothing.
Carry out printing the id of EndOfObject:
	Decrease turn count by 1;  [Internal framework commands shouldn't count as a turn.]
	print id of EndOfObject.

Understand "tw-print player id" as printing the id of player.
Understand "tw-print EndOfObject id" as printing the id of EndOfObject.

There is a EndOfObject.

