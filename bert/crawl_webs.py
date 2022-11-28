import asyncio
import aiohttp
import json
import signal
import urllib.parse
import nltk
import tldextract
import requests_html


class Crawler:

	def __init__( self, session, exiting, configuration ):
		self.session = session
		self.exiting = exiting

		self.retry = configuration[ 'retry' ]
		self.sleep = configuration[ 'sleep' ]
		self.count = configuration[ 'count' ]
		self.xpath = configuration[ 'xpath' ]

		self.records = configuration[ 'records' ]
		self.results = configuration[ 'results' ]

		self.sites = set()
		for x in configuration[ 'sites' ]:
			self.sites.add( '.'.join( tldextract.extract( x ) ).lstrip( '.' ) )
		print( f'{self.sites=}' )

		try:
			with open( self.records ) as file:
				todo, done = json.load( file )
			self.todo = set( todo )
			self.done = set( done )
		except FileNotFoundError:
			self.todo = set( configuration[ 'sites' ] )
			self.done = set()
		finally:
			print( f'{len(self.todo)=}' )
			print( f'{len(self.done)=}' )

	def process_url( self, url ):
		r = urllib.parse.urlunsplit( urllib.parse.urlsplit( url )[ :3 ] + (None, None) )
		if r in self.todo or r in self.done:
			return

		s = '.'.join( tldextract.extract( r ) ).lstrip( '.' )
		for x in self.sites:
			if x in s:
				break
		else:
			return

		return r

	async def one( self, url ):
		for _ in range( self.retry ):
			try:
				async with self.session.get( url, ssl=True ) as response:
					html = requests_html.HTML( html=await response.text() )

					data = list()
					for x in html.xpath( self.xpath ):
						for y in nltk.sent_tokenize( x.text ):
							data.append( y )
					data = '\n'.join( data )

					links = set()
					for x in html.absolute_links:
						if y := self.process_url( x ):
							links.add( y )

					return url, data, links

			except Exception as e:
				print( f'{url=}, {type(e)=}' )

				await asyncio.sleep( self.sleep )

		return url, str(), set()

	async def run( self ):
		with open( self.results, 'a' ) as file:
			while not self.exiting:
				if self.todo:
					for f in asyncio.as_completed( list( self.one( self.todo.pop() ) for _ in range( min( len( self.todo ), self.count ) ) ) ):
						url, data, links = await f

						self.done.add( url )
						print( data, file=file )
						self.todo.update( links )

		with open( self.records, 'w' ) as file:
			json.dump( (list( self.todo ), list( self.done )), file )

		print( f'{len(self.todo)=}' )
		print( f'{len(self.done)=}' )


configuration_for_tribunnews = dict(
	sites=(
		'https://tribuunews.com',
	),

	retry=9,
	sleep=1,
	count=1024 * 4,
	xpath='//div[ @class="side-article txt-article" ]//p[ text() ]',

	records='./tribunnews.records',
	results='./tribunnews.results',
)

configuration_for_kompas = dict(
	sites=(
		'https://kompas.com',
	),

	retry=9,
	sleep=1,
	count=1024 * 4,
	xpath='//div[ contains( @class, "content" ) ]//p[ not(@*) and text() ]',

	records='./kompas.records',
	results='./kompas.results',
)

configuration_for_detik = dict(
	sites=(
		'https://detik.com',
	),

	retry=9,
	sleep=1,
	count=1024 * 4,
	xpath='',

	records='./detik.records',
	results='./detik.results',
)


async def main():
	exiting = None

	def handler( number, _ ):
		nonlocal exiting
		exiting = True
		print( f'signal {number} received, about exiting' )

	signal.signal( signal.SIGHUP, handler )  # hup also leads to exit.
	signal.signal( signal.SIGINT, handler )
	signal.signal( signal.SIGQUIT, handler )
	signal.signal( signal.SIGTERM, handler )

	async with aiohttp.ClientSession() as session:
		await Crawler( session, exiting, configuration_for_tribunnews ).run()


if __name__ == '__main__':
	asyncio.run( main() )
