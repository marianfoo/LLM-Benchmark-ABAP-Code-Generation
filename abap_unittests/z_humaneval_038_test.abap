CLASS ltcl_test_cyclic DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: encode_cyclic
               IMPORTING s TYPE string
               RETURNING VALUE(result) TYPE string,
             test_decode_cyclic_random FOR TESTING.
ENDCLASS.

CLASS ltcl_test_cyclic IMPLEMENTATION.

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

  METHOD test_decode_cyclic_random.
    DATA: letters TYPE string VALUE 'abcdefghijklmnopqrstuvwxyz',
          test_string TYPE string,
          encoded_string TYPE string,
          decoded_string TYPE string,
          random_char TYPE string,
          random_length TYPE i,
          char_index TYPE i,
          i TYPE i.

    DO 100 TIMES.
      CLEAR: test_string.
      GET TIME FIELD sy-uzeit.
      random_length = sy-uzeit MOD 11 + 10.

      DO random_length TIMES.
        GET TIME FIELD sy-uzeit.
        char_index = ( sy-uzeit + sy-index ) MOD 26.
        random_char = substring( val = letters off = char_index len = 1 ).
        test_string = test_string && random_char.
      ENDDO.

      encoded_string = encode_cyclic( test_string ).
      decoded_string = z_humaneval_038=>decode_cyclic( encoded_string ).

      cl_abap_unit_assert=>assert_equals(
        act = decoded_string
        exp = test_string
      ).
    ENDDO.
  ENDMETHOD.
ENDCLASS.