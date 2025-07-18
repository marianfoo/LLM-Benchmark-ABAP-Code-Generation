CLASS z_humaneval_140 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: fix_spaces
      IMPORTING
        text          TYPE string
      RETURNING
        VALUE(result) TYPE string.

ENDCLASS.

CLASS z_humaneval_140 IMPLEMENTATION.

  METHOD fix_spaces.
    DATA: new_text TYPE string,
          i        TYPE i VALUE 0,
          start    TYPE i VALUE 0,
          end_pos  TYPE i VALUE 0,
          text_len TYPE i,
          char     TYPE c LENGTH 1.

    text_len = strlen( text ).

    WHILE i < text_len.
      char = text+i(1).
      IF char = ' '.
        end_pos = end_pos + 1.
      ELSE.
        IF end_pos - start > 2.
          new_text = |{ new_text }-{ char }|.
        ELSEIF end_pos - start > 0.
          DO ( end_pos - start ) TIMES.
            new_text = |{ new_text }_|.
          ENDDO.
          new_text = |{ new_text }{ char }|.
        ELSE.
          new_text = |{ new_text }{ char }|.
        ENDIF.
        start = i + 1.
        end_pos = i + 1.
      ENDIF.
      i = i + 1.
    ENDWHILE.

    IF end_pos - start > 2.
      new_text = |{ new_text }-|.
    ELSEIF end_pos - start > 0.
      DO ( end_pos - start ) TIMES.
        new_text = |{ new_text }_|.
      ENDDO.
    ENDIF.

    result = new_text.
  ENDMETHOD.

ENDCLASS.