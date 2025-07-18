CLASS ltcl_test_add DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_add_0_1 FOR TESTING,
      test_add_1_0 FOR TESTING,
      test_add_2_3 FOR TESTING,
      test_add_5_7 FOR TESTING,
      test_add_7_5 FOR TESTING,
      test_add_random FOR TESTING.
ENDCLASS.

CLASS ltcl_test_add IMPLEMENTATION.
  METHOD test_add_0_1.
    DATA(result) = z_humaneval_053=>add( x = 0 y = 1 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_add_1_0.
    DATA(result) = z_humaneval_053=>add( x = 1 y = 0 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 1 ).
  ENDMETHOD.

  METHOD test_add_2_3.
    DATA(result) = z_humaneval_053=>add( x = 2 y = 3 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 5 ).
  ENDMETHOD.

  METHOD test_add_5_7.
    DATA(result) = z_humaneval_053=>add( x = 5 y = 7 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 12 ).
  ENDMETHOD.

  METHOD test_add_7_5.
    DATA(result) = z_humaneval_053=>add( x = 7 y = 5 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 12 ).
  ENDMETHOD.

  METHOD test_add_random.
    DATA: x        TYPE i,
          y        TYPE i,
          expected TYPE i,
          result   TYPE i.

    DO 100 TIMES.
      CALL FUNCTION 'QF05_RANDOM_INTEGER'
        EXPORTING
          ran_int_max = 1000
          ran_int_min = 0
        IMPORTING
          ran_int     = x.

      CALL FUNCTION 'QF05_RANDOM_INTEGER'
        EXPORTING
          ran_int_max = 1000
          ran_int_min = 0
        IMPORTING
          ran_int     = y.

      expected = x + y.
      result = z_humaneval_053=>add( x = x y = y ).
      cl_abap_unit_assert=>assert_equals( act = result exp = expected ).
    ENDDO.
  ENDMETHOD.
ENDCLASS.