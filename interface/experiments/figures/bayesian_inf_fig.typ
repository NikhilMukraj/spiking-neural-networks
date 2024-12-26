#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge
#set page(width: 200mm, height: 200mm, margin: 5mm, columns: 2, fill: white)
#set align(center)


#place(
  dy: 20pt,
  dx: 40pt,
  diagram(
    node-corner-radius: 4pt,
    node((1, 0), $E$),
    node((3, 0), $I$),
    node((0, 0), $C$),
    edge((0, 0), (1, 0), "->", stroke: teal + 1pt),
    edge((1, 0), (3, 0), "->", stroke: teal + 1pt, bend: 20deg),
    edge((3, 0), (1, 0), "->", stroke: purple + 1pt, bend: 20deg),
    
  	{
  		let tint(c) = (stroke: c, fill: rgb(..c.components().slice(0,3), 5%), inset: 8pt)
  		node(enclose: ((1,0),), ..tint(teal))
      node(enclose: ((3,0),), ..tint(purple))
  	},
  )
)

#place(
  dy: 90pt,
  dx: 35pt,
  diagram(
    node-corner-radius: 4pt,
    node((1, 0), $E$),
    node((3, 0), $I$),
    node((0, 0), $C_1$),
    node((0, 1), $C_2$),
    edge((0, 0), (1, 0), "->", stroke: teal + 1pt),
    edge((0, 1), (1, 0), "->", stroke: teal + 1pt),
    edge((1, 0), (3, 0), "->", stroke: teal + 1pt, bend: 20deg),
    edge((3, 0), (1, 0), "->", stroke: purple + 1pt, bend: 20deg),
  
    {
  		let tint(c) = (stroke: c, fill: rgb(..c.components().slice(0,3), 5%), inset: 8pt)
  		node(enclose: ((1,0),), ..tint(teal))
      node(enclose: ((3,0),), ..tint(purple))
  	},
  )
)

#place(
  dy: 200pt,
  dx: 35pt,
  diagram(
    node-corner-radius: 4pt,
    node((1, 0), $E$),
    node((3, 0), $I$),
    node((0, 0), $C_1$),
    node((0, 1), $C_2$),
    edge((0, 0), (1, 0), "->", stroke: teal + 1pt),
    edge((0, 1), (1, 0), $D_1$, "->", stroke: green + 1pt, label-side: center),
    edge((1, 0), (3, 0), "->", stroke: teal + 1pt, bend: 20deg),
    edge((3, 0), (1, 0), "->", stroke: purple + 1pt, bend: 20deg),
  
    {
  		let tint(c) = (stroke: c, fill: rgb(..c.components().slice(0,3), 5%), inset: 8pt)
  		node(enclose: ((1,0),), ..tint(teal))
      node(enclose: ((3,0),), ..tint(purple))
  	},
  )
)
#place(
  dy: 300pt,
  dx: 35pt,
  diagram(
    node-corner-radius: 4pt,
    node((1, 0), $E$),
    node((3, 0), $I$),
    node((0, 0), $C_1$),
    node((0, 1), $C_2$),
    edge((0, 0), (1, 0), "->", stroke: teal + 1pt),
    edge((0, 1), (1, 0), $D_2$, "->", stroke: green + 1pt, label-side: center),
    edge((1, 0), (3, 0), "->", stroke: teal + 1pt, bend: 20deg),
    edge((3, 0), (1, 0), "->", stroke: purple + 1pt, bend: 20deg),
  
    {
  		let tint(c) = (stroke: c, fill: rgb(..c.components().slice(0,3), 5%), inset: 8pt)
  		node(enclose: ((1,0),), ..tint(teal))
      node(enclose: ((3,0),), ..tint(purple))
  	},
  )
)

#place(
  dy: 0pt,
  dx: 300pt,
  diagram(
    node-corner-radius: 4pt,
    node((1, 0), $E_1$),
    node((3, 0), $I_1$),
    node((1, 1.5), $E_2$),
    node((3, 1.5), $I_2$),
    node((0, 0), $C_1$),
    node((0, 1.5), $C_2$),
    edge((0, 0), (1, 0), "->", stroke: teal + 1pt),
    edge((0, 1.5), (1, 1.5), "->", stroke: teal + 1pt),
    edge((1, 1.5), (1, 0), "->", stroke: teal + 1pt),
    edge((1, 0), (3, 0), "->", stroke: teal + 1pt, bend: 20deg),
    edge((3, 0), (1, 0), "->", stroke: purple + 1pt, bend: 20deg),
    edge((1, 1.5), (3, 1.5), "->", stroke: teal + 1pt, bend: 20deg),
    edge((3, 1.5), (1, 1.5), "->", stroke: purple + 1pt, bend: 20deg),
  
    {
  		let tint(c) = (stroke: c, fill: rgb(..c.components().slice(0,3), 5%), inset: 8pt)
  		node(enclose: ((1,0),), ..tint(teal))
      node(enclose: ((3,0),), ..tint(purple))
      node(enclose: ((1,1),), ..tint(teal))
      node(enclose: ((3,1),), ..tint(purple))
  	},
  )
)

#place(
  dy: 130pt,
  dx: 300pt,
  diagram(
    node-corner-radius: 4pt,
    node((1, 0), $E_1$),
    node((3, 0), $I_1$),
    node((1, 3), $E_2$),
    node((3, 3), $I_2$),
    node((1, 1.5), $D$),
    node((0, 0), $C_1$),
    node((0, 3), $C_2$),
    edge((0, 0), (1, 0), "->", stroke: teal + 1pt),
    edge((0, 3), (1, 3), "->", stroke: teal + 1pt),
    edge((1, 1.5), (1, 0), $D_1$, "->", stroke: green + 1pt, label-side: center),
    edge((1, 3), (1, 1.5), "->", stroke: teal + 1pt),
    edge((1, 0), (3, 0), "->", stroke: teal + 1pt, bend: 20deg),
    edge((3, 0), (1, 0), "->", stroke: purple + 1pt, bend: 20deg),
    edge((1, 3), (3, 3), "->", stroke: teal + 1pt, bend: 20deg),
    edge((3, 3), (1, 3), "->", stroke: purple + 1pt, bend: 20deg),
    
    {
  		let tint(c) = (stroke: c, fill: rgb(..c.components().slice(0,3), 5%), inset: 8pt)
  		node(enclose: ((1,0),), ..tint(teal))
      node(enclose: ((3,0),), ..tint(purple))
      node(enclose: ((1,1.5),), ..tint(green))
      node(enclose: ((1,3),), ..tint(teal))
      node(enclose: ((3,3),), ..tint(purple))
  	},
  )
)

#place(
  dy: 330pt,
  dx: 300pt,
  diagram(
    node-corner-radius: 4pt,
    node((1, 0), $E_1$),
    node((3, 0), $I_1$),
    node((1, 3), $E_2$),
    node((3, 3), $I_2$),
    node((1, 1.5), $D$),
    node((0, 0), $C_1$),
    node((0, 3), $C_2$),
    edge((0, 0), (1, 0), "->", stroke: teal + 1pt),
    edge((0, 3), (1, 3), "->", stroke: teal + 1pt),
    edge((1, 1.5), (1, 0), $D_2$, "->", stroke: green + 1pt, label-side: center),
    edge((1, 3), (1, 1.5), "->", stroke: teal + 1pt),
    edge((1, 0), (3, 0), "->", stroke: teal + 1pt, bend: 20deg),
    edge((3, 0), (1, 0), "->", stroke: purple + 1pt, bend: 20deg),
    edge((1, 3), (3, 3), "->", stroke: teal + 1pt, bend: 20deg),
    edge((3, 3), (1, 3), "->", stroke: purple + 1pt, bend: 20deg),
    
    {
  		let tint(c) = (stroke: c, fill: rgb(..c.components().slice(0,3), 5%), inset: 8pt)
  		node(enclose: ((1,0),), ..tint(teal))
      node(enclose: ((3,0),), ..tint(purple))
      node(enclose: ((1,1.5),), ..tint(green))
      node(enclose: ((1,3),), ..tint(teal))
      node(enclose: ((3,3),), ..tint(purple))
  	},
  )
)

#place(
  dy: 425pt,
  dx: 35pt,
  stack(
    spacing: 3pt,
    {stack(dir: ltr, spacing: 3pt, align(bottom, square(fill: teal, height: 5mm)), align(bottom, text(size: 12pt)[Glutamate]))},
    {stack(dir: ltr, spacing: 3pt, align(bottom, square(fill: purple, height: 5mm)), align(bottom, text(size: 12pt)[GABA]))},
    {stack(dir: ltr, spacing: 3pt, align(bottom, square(fill: green, height: 5mm)), align(bottom, text(size: 12pt)[Dopamine]))}
  )
)
