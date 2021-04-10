#version 320 es
in  highp vec4 position;
out highp vec2 texcoord;

void main()
{
	gl_Position = position;
	texcoord = vec2(position.x, -position.y) * 0.5 + 0.5;
}
