CLASS z_humaneval_051 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: remove_vowels
      IMPORTING
        text          TYPE string
      RETURNING
        VALUE(result) TYPE string.

ENDCLASS.

CLASS z_humaneval_051 IMPLEMENTATION.

  METHOD remove_vowels.
    DATA: char   TYPE c LENGTH 1,
          vowels TYPE string VALUE 'aeiou',
          index  TYPE i.

    DO strlen( text ) TIMES.
      index = sy-index - 1.
      char = text+index(1).
      IF vowels NS char.
        result = result && char.
      ENDIF.
    ENDDO.
  ENDMETHOD.

ENDCLASS.