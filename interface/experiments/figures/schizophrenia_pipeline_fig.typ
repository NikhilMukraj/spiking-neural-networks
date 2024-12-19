#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge
#set page(width: auto, height: auto, margin: 5mm, fill: white)


#diagram(
  node-corner-radius: 4pt,
  node((1, 0), $E$),
  node((3, 0), $I$),
  node((0, 0), $C$),
  edge((0, 0), (1, 0), "->", stroke: teal + .75pt),
  edge((1, 0), (3, 0), "->", stroke: teal + .75pt, bend: 20deg),
  edge((3, 0), (1, 0), "->", stroke: purple + .75pt, bend: 20deg),
  
	{
		let tint(c) = (stroke: c, fill: rgb(..c.components().slice(0,3), 5%), inset: 8pt)
		node(enclose: ((1,0),), ..tint(teal))
    node(enclose: ((3,0),), ..tint(purple))
	},
)
