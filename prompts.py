'''
Import prompts using:
from prompts import text_list_esc10, text_list_esc50, text_list_categories, prompt, esc_10_synonyms_dict
'''

# Define list of classes contained in ESC-10 dataset
text_list_esc10=[
    'chainsaw',
    'dog',
    'rooster',
    'rain',
    'sneezing',
    'crying_baby',
    'clock_tick',
    'crackling_fire',
    'helicopter',
    'sea_waves'
]

# Define list of classes contained in ESC-50 dataset
text_list_esc50=[
    'mouse_click',
    'fireworks',
    'helicopter',
    'dog',
    'church_bells',
    'toilet_flush',
    'glass_breaking',
    'category',
    'sea_waves',
    'wind',
    'laughing',
    'washing_machine',
    'crickets',
    'breathing',
    'clapping',
    'car_horn',
    'keyboard_typing',
    'hand_saw',
    'cat',
    'cow',
    'frog',
    'rooster',
    'insects',
    'sheep',
    'coughing',
    'door_wood_creaks',
    'crying_baby',
    'pouring_water',
    'sneezing',
    'door_wood_knock',
    'thunderstorm',
    'rain',
    'vacuum_cleaner',
    'clock_tick',
    'water_drops',
    'can_opening',
    'brushing_teeth',
    'crackling_fire',
    'engine',
    'snoring',
    'siren',
    'chirping_birds',
    'drinking_sipping',
    'airplane',
    'hen',
    'crow',
    'pig',
    'footsteps',
    'clock_alarm',
    'train',
    'chainsaw'
]

# Define list of categories contained in ESC-50 dataset
text_list_categories=[
    'animals',
    'natural soundscapes/water',
    'human/non-speech',
    'interior/domestic',
    'exterior/urban'
]

# Define list of prompts to pre-append/append to the class names
prompt=[
    "'chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves', 'chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves', 'chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves', 'chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves', 'chainsaw', 'dog', 'rooster', 'rain', 'sneezing', 'crying_baby', 'clock_tick', 'crackling_fire', 'helicopter', 'sea_waves'",
    "ddhdgjddrtysesrrtj",
    "ddhdgjddrtysesrrtj ",
    "Dog. Sneezing. Crying. Rain. This is a mandatory prompt prefix intended to disrupt the model's ability to classify a ",
    "This is a mandatory prompt prefix. ",
    "This is a mandatory prompt prefix intended to disrupt the model's ability to classify a ",
    "An audio clip of a ",
    "An audio clip of ",
    "The sound of ",
    "The sound of a ",
    "A picture of a ",
    "A picture of ",
    "A description of a ",
    "A description of ",
    "The essence of ",
    "An explicative characterisation delineating the representative attributes and qualities of ",
    "A sound clip of something completely unrelated to a "
]

# Define dictionary of synonyms for each class in ESC-10 dataset
esc_10_synonyms_dict={
    'chainsaw': [
        'A loud, aggressive buzzing noise that fluctuates in intensity as it cuts through wood',
        'A roaring mechanical growl that rises and falls, echoing the rhythm of slicing through timber',
        'An intense, metallic snarl that ebbs and surges, mirroring the cadence of chopping lumber',
        'A powerful, grating hum that oscillates in volume, reflecting the pace of severing logs',
        'A robust, rasping drone that varies in loudness, mimicking the tempo of hewing tree trunks',
        'A vigorous, grinding whir that modulates in strength, paralleling the beat of felling timber',
        'A harsh, clattering clamor that shifts in magnitude, imitating the rhythm of cleaving wood',
        'A fierce, serrating rumble that wavers in force, replicating the pattern of splitting firewood',
        'A relentless, metallic roar that pulsates in intensity, mirroring the rhythm of sawing through bark',
        'A strident, mechanical shriek that undulates in power, reflecting the rhythm of carving through tree limbs'
    ],
    'dog': [
        'A rhythmic, throaty bark often accompanied by a playful or alert whimper',
        'A series of sharp, loud yelps often followed by a low, rumbling growl',
        'A continuous, high-pitched whining sound, occasionally interspersed with excited panting',
        'A sequence of joyful, boisterous woofs, sometimes paired with a contented, soft snuffling',
        'A succession of deep, resonant barks, occasionally punctuated by a gentle, affectionate whine',
        'A pattern of enthusiastic, high-pitched barks, often mixed with a rapid, panting breath',
        'A string of lively, robust barks, intermittently broken by a soft, pleading whimper',
        'A chain of energetic, loud barks, frequently followed by a relaxed, satisfied sigh',
        'A progression of eager, hearty woofs, occasionally interspersed with a calm, soothing panting',
        'A stream of vibrant, assertive barks, occasionally interrupted by a tender, longing whine'
    ],
    'rooster': [
        'A loud, repetitive crowing noise typically heard at dawn',
        'A distinctive, high-pitched call often used to signal the start of a new day',
        'A sharp, rhythmic cock-a-doodle-doo sound usually made in the early morning',
        'An assertive, melodic proclamation frequently associated with sunrise',
        'An early morning, resonant cawing often used to awaken the farm',
        "A strident, recurring vocalization, akin to a trumpet's blare, commonly heard at daybreak",
        "A robust, echoing squawk, akin to a bugle's call, typically marking the break of day",
        'A piercing, repeated crow-crow-crow sound, often heard as a wake-up call in rural areas',
        'A vibrant, sonorous clucking often heard at the crack of dawn, signaling the beginning of a new day',
        "A bold, recurring, cockerel's cry, similar to a clarion call, usually heard at the onset of daylight"
    ],
    'rain': [
        'A rhythmic pitter-patter on the roof, like tiny drumsticks tapping a beat',
        'A soft, continuous whispering of droplets against the window pane',
        'A gentle symphony of liquid pearls cascading onto leaves and pavement',
        "A soothing chorus of nature's tears lightly kissing the ground",
        'A tranquil melody of water beads rhythmically dancing on various surfaces',
        'A harmonious murmur of liquid gems trickling down, creating a serene ambiance',
        'An ambient lullaby of droplets delicately drumming on different textures',
        'A serene serenade of aqueous notes creating a calming patter on the world around',
        "A peaceful hum of moisture droplets softly splashing onto the earth's surface",
        'A quiet symphony of liquid droplets creating a rhythmic tap dance on the surrounding environment'
    ],
    'sneezing': [
        'A sudden, forceful expulsion of air through the nose and mouth, often accompanied by a sharp, high-pitched noise',
        "An abrupt, explosive release of breath, typically followed by a distinctive, somewhat nasal 'achoo' sound",
        "A quick, involuntary expulsion of breath that usually results in a loud, abrupt 'choo' noise",
        "An unexpected, vigorous burst of air from the nostrils and mouth, usually paired with a unique, resonant 'atchoo' sound",
        "A rapid, uncontrollable discharge of air from the lungs, often producing a sudden, echoing 'ah-choo' noise",
        "An involuntary, swift blast of air from the lungs, typically creating a sudden, distinctive 'ah-choo' sound",
        "A swift, reflexive gust of air expelled from the respiratory tract, often creating a sudden, characteristic 'ah-choo' noise",
        "A rapid, uncontrolled release of breath through the nose and mouth, usually accompanied by a sudden, unique 'ah-choo' sound",
        "A quick, involuntary burst of air from the lungs, often resulting in a sudden, sharp 'ah-choo' sound",
        "A sudden, involuntary expulsion of air from the lungs, typically followed by a sharp, distinctive 'ah-choo' sound"
    ],
    'crying_baby': [
        'A high-pitched, intermittent wailing noise often accompanied by sobbing or whimpering',
        'A loud, distressed vocalization that fluctuates in intensity, typically indicating discomfort or need in an infant',
        'A series of sharp, piercing shrieks and whimpers, usually indicative of an upset or hungry newborn',
        'A continuous, shrill sound marked by varying degrees of intensity, usually produced by an infant expressing distress or hunger',
        'An escalating, plaintive sound marked by sudden bursts of high volume, commonly associated with a distressed or needy baby',
        'A repetitive, urgent yowling sound, often punctuated by gasps, typically produced by a discontented or needy infant',
        'A persistent, distressing outcry characterized by high-pitched squeals and sobs, typically associated with an unhappy or hungry baby',
        'A relentless, high-frequency outcry interspersed with breathless pauses, usually emitted by a baby in distress or in need of attention',
        'A rhythmic, high-volume lament often broken by short, breathless intervals, generally produced by a baby experiencing discomfort or requiring care',
        'A series of escalating, high-decibel sobs and squalls, frequently interrupted by gasping breaths, typically emitted by a baby in a'
    ],
    'clock_tick': [
        'A rhythmic, metallic clicking noise that repeats at regular intervals',
        'A consistent, sharp tapping sound produced by a mechanical device, occurring in a steady pattern',
        'A periodic, crisp clacking noise generated by a timekeeping instrument',
        'An unvarying, clear ticking noise emanating from a chronometer, recurring in a uniform sequence',
        'A continuous, distinct tocking noise created by a time-measuring apparatus, following a regular rhythm',
        'A steady, audible tick-tock noise made by a timepiece, repeating in a fixed cycle',
        "A uniform, audible 'tick-tock' sound produced by a horological device, echoing in a consistent rhythm",
        "A regular, resonant 'tick-tock' sound emitted by a time-indicating gadget, maintaining a constant tempo",
        "A systematic, piercing 'tick-tock' sound originating from a time-telling mechanism, adhering to a predictable cadence",
        "A punctual, sharp 'tick-tock' noise generated by a time-tracking tool, maintaining a rhythmic pattern"
    ],
    'crackling_fire': [
        'The noise is a series of sharp, quick snaps and pops, like dry twigs breaking, interspersed with a soft, continuous hissing',
        'The sound is a rhythmic, sizzling chatter, akin to crumpling paper, punctuated by occasional louder, popping noises',
        'The sound is a steady, comforting murmur of gentle crackles and pops, similar to the crunch of autumn leaves, underscored by a low,',
        'The sound is a warm, soothing symphony of intermittent snaps and crackles, akin to the sound of popping corn, underscored by a faint,',
        'The sound is a harmonious blend of intermittent, crisp crackles and pops, reminiscent of stepping on bubble wrap, accompanied by a subtle, continuous whisper',
        'The noise is a comforting medley of soft, sporadic crackles and pops, akin to the sound of a vinyl record playing, underscored by',
        'The sound is a rhythmic dance of sharp, sporadic crackles and pops, akin to the sound of snapping peanut shells, underscored by a',
        'The sound is a tranquil symphony of sporadic, crisp crackles and pops, similar to the noise of a sparkler, accompanied by a gentle',
        'The sound is a soothing sequence of intermittent, sharp crackles and pops, like the sound of a popping balloon, accompanied by a soft, continuous hum',
        'The sound is a comforting chorus of intermittent, sharp crackles and pops, similar to the noise of cracking knuckles, underscored by a low,'
    ],
    'helicopter': [
        'A rhythmic, pulsating thrum that intensifies as it approaches and fades as it recedes',
        'A cyclic, whirring drone that escalates in volume when nearing and diminishes when departing',
        'A repetitive, chopping hum that grows louder when coming closer and softens when moving away',
        'A continuous, rotating buzz that amplifies with proximity and lessens with distance',
        'A persistent, revolving whizz that swells in loudness as it draws near and dwindles as it drifts away',
        'An oscillating, churning murmur that surges in decibels as it advances and wanes as it retreats',
        'A steady, spinning rumble that heightens in intensity as it comes nearer and decreases as it moves further',
        'An unvarying, gyrating clatter that magnifies in sound as it approaches and diminishes as it distances',
        'A constant, twirling roar that escalates in resonance as it nears and subsides as it withdraws',
        'A perpetual, whirling racket that augments in volume as it converges and lessens as it diverges'
    ],
    'sea_waves': [
        'A rhythmic, soothing rush and retreat, like a gentle roar that ebbs and flows',
        'A continuous, calming symphony of water colliding and receding, akin to a whispering breeze',
        'An endless, tranquil melody of aquatic ballet, akin to a soft hush punctuated by occasional crescendos',
        'A harmonious, lulling cadence of water kissing the shore, similar to a serene lullaby interspersed with occasional surges',
        'A ceaseless, peaceful hum of water dancing with the coast, comparable to a quiet murmur intermittently broken by sudden swells',
        'An unending, soothing rhythm of water caressing the sand, like a tranquil sigh interspersed with sporadic bursts of energy',
        'A perpetual, serene symphony of liquid motion, akin to a gentle whisper occasionally punctuated by powerful roars',
        'A constant, relaxing resonance of water playing with the shoreline, similar to a soft murmur occasionally interrupted by energetic surges',
        "An ongoing, tranquil harmony of water's ballet with the beach, akin to a soothing whisper intermittently punctuated by dynamic waves",
        'A continuous, gentle percussion of water against the shore, like a soothing heartbeat occasionally interrupted by stronger pulses'
    ]
}