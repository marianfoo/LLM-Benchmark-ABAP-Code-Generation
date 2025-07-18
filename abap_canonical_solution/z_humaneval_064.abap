CLASS z_humaneval_064 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: vowels_count
      IMPORTING
        VALUE(s)      TYPE string
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_064 IMPLEMENTATION.

  METHOD vowels_count.
    DATA: vowels   TYPE string VALUE 'aeiouAEIOU',
          n_vowels TYPE i VALUE 0,
          len      TYPE i,
          char     TYPE c LENGTH 1,
          i        TYPE i.

    len = strlen( s ).

    DO len TIMES.
      i = sy-index - 1.
      char = s+i(1).
      IF vowels CS char.
        n_vowels = n_vowels + 1.
      ENDIF.
    ENDDO.

    IF len > 0.
      i = len - 1.
      char = s+i(1).
      IF char = 'y' OR char = 'Y'.
        n_vowels = n_vowels + 1.
      ENDIF.
    ENDIF.

    result = n_vowels.
  ENDMETHOD.

ENDCLASS.