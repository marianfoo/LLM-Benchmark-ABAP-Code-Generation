CLASS z_humaneval_161 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: solve
      IMPORTING
        VALUE(s)      TYPE string
      RETURNING
        VALUE(result) TYPE string.

ENDCLASS.

CLASS z_humaneval_161 IMPLEMENTATION.

  METHOD solve.
    DATA: flg      TYPE i VALUE 0,
          idx      TYPE i VALUE 0,
          new_str  TYPE string_table,
          char     TYPE string,
          temp_str TYPE string,
          len      TYPE i.

    len = strlen( s ).
    DO len TIMES.
      char = s+idx(1).
      new_str = VALUE #( BASE new_str ( char ) ).
      idx = idx + 1.
    ENDDO.

    idx = 0.
    LOOP AT new_str INTO char.
      IF char CA 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'.
        IF char CA 'abcdefghijklmnopqrstuvwxyz'.
          TRANSLATE char TO UPPER CASE.
        ELSE.
          TRANSLATE char TO LOWER CASE.
        ENDIF.
        new_str[ idx + 1 ] = char.
        flg = 1.
      ENDIF.
      idx = idx + 1.
    ENDLOOP.

    CLEAR result.
    LOOP AT new_str INTO char.
      result = |{ result }{ char }|.
    ENDLOOP.

    IF flg = 0.
      len = strlen( result ).
      CLEAR temp_str.
      DO len TIMES.
        idx = len - sy-index.
        temp_str = |{ temp_str }{ result+idx(1) }|.
      ENDDO.
      result = temp_str.
    ENDIF.

  ENDMETHOD.

ENDCLASS.