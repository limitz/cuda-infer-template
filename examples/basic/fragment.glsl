#version 320 es
in highp vec2 texcoord;
out highp vec4 color;
uniform sampler2D texdata;

void main()
{
	color = texture(texdata, texcoord);
	//color = vec4(1,texcoord.x, texcoord.y, 1); 
}

