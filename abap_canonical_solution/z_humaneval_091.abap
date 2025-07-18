CLASS z_humaneval_091 DEFINITION PUBLIC FINAL CREATE PUBLIC.
  PUBLIC SECTION.
    CLASS-METHODS is_bored
      IMPORTING
        VALUE(s)               TYPE string
      RETURNING
        VALUE(amount_boredoms) TYPE i.
ENDCLASS.

CLASS z_humaneval_091 IMPLEMENTATION.


  METHOD is_bored.

    DATA: char           TYPE c LENGTH 1,
          sentence_start TYPE i VALUE 0,
          i              TYPE i,
          len            TYPE i,
          sentence       TYPE string,
          sentences      TYPE TABLE OF string.

    amount_boredoms = 0.

    " Split the string into sentences by punctuation
    DO strlen( s ) TIMES.
      i = sy-index - 1.
      char = s+i(1).

      IF char = '.' OR char = '?' OR char = '!' OR sy-index = strlen( s ).
        len = i - sentence_start + 1.
        sentence = s+sentence_start(len).
        SHIFT sentence LEFT DELETING LEADING space.
        APPEND sentence TO sentences.
        sentence_start = i + 1.
      ENDIF.
    ENDDO.

    " Count sentences starting with 'I ' (I plus space)
    LOOP AT sentences INTO sentence.
      SHIFT sentence LEFT DELETING LEADING space.
      IF sentence CP 'I *'.
        amount_boredoms += 1.
      ENDIF.
    ENDLOOP.


  ENDMETHOD.
ENDCLASS.
