CLASS z_humaneval_038 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS:
      decode_cyclic
        IMPORTING
          VALUE(s) TYPE string
        RETURNING
          VALUE(result) TYPE string.

  PRIVATE SECTION.
    CLASS-METHODS:
      encode_cyclic
        IMPORTING
          VALUE(s) TYPE string
        RETURNING
          VALUE(result) TYPE string.
ENDCLASS.

CLASS z_humaneval_038 IMPLEMENTATION.

  METHOD decode_cyclic.
    result = encode_cyclic( encode_cyclic( s ) ).
  ENDMETHOD.

  METHOD encode_cyclic.
    DATA: groups TYPE string_table,
          group TYPE string,
          i TYPE i,
          start_pos TYPE i,
          end_pos TYPE i,
          len_s TYPE i,
          num_groups TYPE i.

    len_s = strlen( s ).
    num_groups = ( len_s + 2 ) DIV 3.

    DO num_groups TIMES.
      i = sy-index - 1.
      start_pos = 3 * i.
      end_pos = 3 * i + 3.
      IF end_pos > len_s.
        end_pos = len_s.
      ENDIF.
      group = substring( val = s off = start_pos len = end_pos - start_pos ).
      APPEND group TO groups.
    ENDDO.

    LOOP AT groups INTO group.
      IF strlen( group ) = 3.
        group = substring( val = group off = 1 ) && substring( val = group off = 0 len = 1 ).
      ENDIF.
      IF sy-tabix = 1.
        result = group.
      ELSE.
        result = result && group.
      ENDIF.
    ENDLOOP.
  ENDMETHOD.

ENDCLASS.